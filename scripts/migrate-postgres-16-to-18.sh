#!/usr/bin/env bash
# Copy a Ragbot PostgreSQL 16 database into the PostgreSQL 18 volume.
#
# docker-compose.yml moved from pgvector/pgvector:pg16 to pg18 on a new
# volume, because a pg16 data directory cannot be opened by pg18. This script
# copies the old database across with pg_dump/pg_restore and checks that every
# table arrived with the same row count. It never modifies the old volume;
# remove that yourself once you are satisfied (`docker volume rm <name>`).
#
# Usage, from the repository root, before starting ragbot-api on pg18:
#
#   scripts/migrate-postgres-16-to-18.sh [old-volume]
#
# old-volume defaults to <compose project>_ragbot-postgres-data, the volume
# the pg16 service used. The project name follows Compose's own default (the
# directory name) unless COMPOSE_PROJECT_NAME is set. POSTGRES_USER,
# POSTGRES_DB and POSTGRES_PASSWORD are read the same way docker-compose.yml
# reads them.
set -euo pipefail

project="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"
old_volume="${1:-${project}_ragbot-postgres-data}"
user="${POSTGRES_USER:-ragbot}"
db="${POSTGRES_DB:-ragbot}"
password="${POSTGRES_PASSWORD:-ragbot_dev_password_change_me}"
helper="ragbot-pg16-migrate-$$"
compose=(docker compose -p "$project")

if ! docker volume inspect "$old_volume" >/dev/null 2>&1; then
  echo "No volume named $old_volume; nothing to migrate." >&2
  exit 1
fi

echo "Starting the PostgreSQL 18 service..."
"${compose[@]}" up -d --wait postgres

tables=$("${compose[@]}" exec -T postgres psql -U "$user" -d "$db" -Atc \
  "select count(*) from information_schema.tables where table_schema = 'public'")
if [ "$tables" != "0" ]; then
  echo "The PostgreSQL 18 database already has $tables tables; refusing to overwrite it." >&2
  exit 1
fi

echo "Starting a temporary PostgreSQL 16 container on $old_volume..."
docker run -d --name "$helper" \
  -v "$old_volume":/var/lib/postgresql/data \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -e POSTGRES_USER="$user" -e POSTGRES_DB="$db" -e POSTGRES_PASSWORD="$password" \
  pgvector/pgvector:pg16 >/dev/null
trap 'docker rm -f "$helper" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do
  docker exec "$helper" pg_isready -U "$user" -d "$db" >/dev/null 2>&1 && break
  sleep 1
done
docker exec "$helper" pg_isready -U "$user" -d "$db" >/dev/null

echo "Copying roles..."
# The superuser already exists in the new cluster; its CREATE ROLE line is the
# one expected error, so it is filtered out rather than stopping the restore.
docker exec "$helper" pg_dumpall -U "$user" --roles-only \
  | grep -vE "^CREATE ROLE \"?${user}\"?;" \
  | "${compose[@]}" exec -T postgres psql -U "$user" -d postgres -v ON_ERROR_STOP=1 -q

echo "Copying database $db..."
docker exec "$helper" pg_dump -U "$user" -d "$db" -Fc \
  | "${compose[@]}" exec -T postgres pg_restore -U "$user" -d "$db" --exit-on-error

echo "Comparing row counts..."
count_rows='select string_agg(format($q$select %L as t, count(*) as n from %I.%I$q$, table_name, table_schema, table_name), $u$ union all $u$ order by table_name) from information_schema.tables where table_schema = $s$public$s$ and table_type = $b$BASE TABLE$b$'
old_query=$(docker exec "$helper" psql -U "$user" -d "$db" -Atc "$count_rows")
new_query=$("${compose[@]}" exec -T postgres psql -U "$user" -d "$db" -Atc "$count_rows")
old_counts=$(docker exec "$helper" psql -U "$user" -d "$db" -AtF ' ' -c "$old_query order by 1")
new_counts=$("${compose[@]}" exec -T postgres psql -U "$user" -d "$db" -AtF ' ' -c "$new_query order by 1")
if [ "$old_counts" != "$new_counts" ]; then
  echo "Row counts differ between PostgreSQL 16 and 18:" >&2
  diff <(echo "$old_counts") <(echo "$new_counts") >&2 || true
  exit 1
fi
echo "$new_counts" | sed 's/^/  /'
echo "Done. Every table matches. The old volume $old_volume is unchanged."
