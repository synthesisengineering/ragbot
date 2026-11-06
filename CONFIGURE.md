Configuration and personalization instructions for RagBot.AI
============================================================

### Configuring RagBot.AI
If you haven't already downloaded and installed RagBot.AI, read the [installation guide](INSTALL.md).

After successfully installing the dependencies, RagBot.AI needs to be configured with API keys.

1.  Create the synthesis-engineering shared config directory:

```bash
mkdir -p ~/.synthesis
```

2.  Create the API keys file:

```bash
cat > ~/.synthesis/keys.yaml << 'EOF'
# Synthesis API Keys (shared across synthesis-engineering products: ragbot, ragenie, etc.)
default:
  anthropic: "sk-ant-your-key-here"
  openai: "sk-your-key-here"
  google: "your-gemini-key-here"
EOF
chmod 600 ~/.synthesis/keys.yaml
```

3.  Edit `~/.synthesis/keys.yaml` with your actual API keys.

* * * * *

Remember, the keys file contains sensitive information such as API keys, so it should never be shared or published. The file is stored in your home directory at `~/.synthesis/` and is NOT part of the ragbot repository.

The legacy location `~/.config/ragbot/keys.yaml` continues to work as a fallback if `~/.synthesis/keys.yaml` does not exist.

### Configuring the vector backend (pgvector)

Ragbot's default vector store is PostgreSQL with the `pgvector` extension. For Docker Compose users, the database container starts automatically (see [README-DOCKER.md](README-DOCKER.md)).

For native CLI use, point ragbot at any reachable Postgres instance:

```bash
# 1. Install pgvector for your Postgres (example: PostgreSQL 16 via Homebrew)
brew install postgresql@16
brew services start postgresql@16
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector && PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make && PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make install

# 2. Create the ragbot database and enable the extension
createuser -s ragbot
createdb -O ragbot ragbot
psql -U ragbot -d ragbot -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Point ragbot at it
export RAGBOT_DATABASE_URL=postgresql://ragbot:CHANGE_ME@localhost:5432/ragbot

# 4. Verify
ragbot db status
```

PostgreSQL with the `pgvector` extension is the only shipped backend. Qdrant
support was removed in v3.5; if you ran an earlier version with
`RAGBOT_VECTOR_BACKEND=qdrant`, reindex your workspaces into pgvector before
upgrading. The `VectorStore` ABC at `synthesis_engine.vectorstore` still
lets substrate consumers plug in alternatives behind the same contract.

### Discovering and indexing Agent Skills

Ragbot reads Agent Skills (directories containing `SKILL.md`) as first-class content. The full directory tree is honoured — `references/**/*.md` and bundled scripts are all indexed and queryable via RAG.

```bash
ragbot skills list                          # show discovered skills
ragbot skills info <skill-name>             # inspect a specific skill
ragbot skills index                         # index every skill into the 'skills' workspace
ragbot skills index --only synthesis-foo    # narrow to one skill
ragbot skills index --force                 # clear and re-index
```

Default discovery roots:

1. `~/.synthesis/skills/`            (shared install for synthesis-engineering tools)
2. `~/.claude/skills/`                (Claude Code private)
3. `~/.claude/plugins/cache/*/skills/` (plugin-installed)
4. Per-workspace roots from `compile-config.yaml` `sources.skills.roots`

When the `skills` workspace has indexed content, `ragbot chat` automatically merges its results with the user's selected workspace. Override per-call with `--no-skills` (opt out) or `--workspace foo --workspace bar` (explicit list).

### Demo mode (v3.2+)

Set `RAGBOT_DEMO=1` (or run `ragbot --demo`) to evaluate ragbot
end-to-end without configuring a workspace, an inheritance chain, or
the user-config files. Demo mode does three things:

1. **Discovery isolation.** The discovery layer returns ONLY the
   bundled `demo/ai-knowledge-demo/` workspace and the bundled
   `demo/skills/ragbot-demo-skill/` skill. Real workspaces declared
   in `~/.synthesis/console.yaml` and any glob-discovered repos under
   `~/workspaces/*/` are invisible while demo mode is on.
2. **Auto-indexing.** On the first demo invocation, ragbot indexes the
   bundled content into the configured vector store under the
   `demo` workspace (and a separate `demo_skills` workspace for the
   bundled skill). Idempotent on subsequent runs.
3. **API + Web UI signal.** `/health` and `/api/config` both return
   `demo_mode: true`. The Web UI renders a yellow banner so screenshots
   captured in demo mode are unmistakably demo. The healthcheck's
   `vector_backend.workspaces` count is filtered to demo-visible
   collections only — your real workspace count won't leak through the
   UI even if other collections exist on the same vector store.

```bash
# Run any subcommand in demo mode:
RAGBOT_DEMO=1 ragbot db status
RAGBOT_DEMO=1 ragbot skills list
RAGBOT_DEMO=1 ragbot chat -p "What is ragbot?"

# Or as a top-level flag:
ragbot --demo db status
ragbot --demo chat -p "What is ragbot?"
```

To exit demo mode, simply unset the env var: `unset RAGBOT_DEMO`. The
auto-indexed `demo` and `demo_skills` workspaces remain in the vector
store and can be cleared with `ragbot db status` followed by manual
cleanup if desired; they won't appear in non-demo discovery.

### LLM backend selection

Ragbot v3.1+ routes every LLM call through a swappable backend interface (`src/ragbot/llm/`). Two backends ship:

```bash
# Default — wraps the LiteLLM SDK; best provider/model coverage.
export RAGBOT_LLM_BACKEND=litellm

# Opt-in — calls Anthropic / OpenAI / google-genai SDKs directly.
# Smaller dependency chain; no third-party gateway.
export RAGBOT_LLM_BACKEND=direct
```

Both backends honour the same provider request rules (OpenAI's `max_completion_tokens`, Anthropic's adaptive thinking with `output_config` effort and no sampling parameters, Gemini's thinking levels). Adding alternatives like Bifrost, Portkey, or OpenRouter is a single new file implementing `LLMBackend`.

### Reasoning / thinking modes

Every cloud model in `engines.yaml` carries a strict reasoning contract: its effort defaults to the catalog value (for example `high` for Claude Fable 5.1 and Sonnet 5.5, `medium` for GPT-6.1 Sol and Gemini 4 Argon, `minimal` for Gemini 3.5 Flash-Lite), and an override must be one of the model's listed levels — an unsupported value is an error rather than a silent substitution. Local models with hybrid thinking (Qwen3.8, Nemotron 3.5 Super) send no thinking parameter unless you set one, which leaves the model's own default. Models without a `thinking` block (Gemma 4, Muse Glimmer, Mistral Medium 3.5) ignore the parameter.

```bash
# Per-call override
ragbot chat --thinking-effort high -p "explain this..."

# Globally
export RAGBOT_THINKING_EFFORT=low

# Disable on a flagship model
ragbot chat --thinking-effort off -p "..."
```

LiteLLM normalises `reasoning_effort` per provider — Claude 4.x receives `thinking={"type": "adaptive"}`, OpenAI receives `reasoning_effort` directly, Gemini receives the corresponding thinking level.

### Running RagBot.AI

1.  View the RagBot.AI help file to see how to use its capabilities:

Command line version rbot.py
```bash
./rbot --help
```

2. Run rbot to execute a prompt including knowledge from a file. (We'll personalize this later with your own data.)
```bash
./rbot -p "What is RagBot.AI?" -d ./README.md
```

```bash
./rbot -p "Why should I use RagBot.AI?" -d ./README.md
```

You can also specify the model which you wish to use:

```bash
./rbot -p "Why should I use RagBot.AI?" -d ./README.md -m gpt-4
```

You can also run RagBot.AI in a web browser locally on your computer:

```bash
./rbot-web
```
![](screenshots/Screenshot%202024-04-10%20at%2010.46.02 PM.png)

Read the [main documentation](README.md) for examples and more information about RagBot.AI.

### Personalizing RagBot.AI

To personalize RagBot.AI and make it reflect your own user preferences, you can follow the steps below:

#### Where to Store Your Ragbot.AI Data

Ragbot.AI uses a dedicated folder to store your custom instructions, datasets, and any saved prompts or conversations. This data may contain personal or sensitive information, so choosing a secure location is crucial.

Here's how to determine the default data storage location for your operating system:

-   macOS/Linux: `~/ragbot-data` (This translates to a folder named `ragbot-data` within your home directory.)
-   Windows: `%USERPROFILE%\ragbot-data` (This translates to a folder named `ragbot-data` within your user profile directory.)

#### Important Considerations for Data Storage

-   Version Control: If you intend to use version control systems like Git to manage your `ragbot-data` folder, it's highly recommended to avoid placing it within a directory that is synchronized with cloud storage services such as iCloud, Dropbox, OneDrive, or similar. Cloud syncing can lead to conflicts and unexpected behavior with version control.

-   Cloud Backups (Optional): If you prefer having cloud-based backups of your Ragbot.AI data and are not using version control, you may choose to place the `ragbot-data` folder within a directory that is synchronized with your preferred cloud storage provider. This ensures your data is backed up and accessible from multiple devices.

#### Organizing Your Ragbot.AI Data

Within the `ragbot-data` folder, it's recommended to create subfolders to organize your custom instructions and datasets:

-   `instructions`: This subfolder will hold your custom instruction files in Markdown format (`.md`). These files provide Ragbot.AI with guidelines and preferences for its responses.
-   `datasets`: This subfolder will contain your dataset files, also in Markdown format. These files offer contextual information and knowledge to Ragbot.AI, allowing it to generate more relevant and informed responses.

Remember: Always prioritize the security and privacy of your Ragbot.AI data by choosing a storage location that aligns with your needs and security preferences.


1.  Set up your personalized custom instuctions for rbot.

Create a `instructions` folder containing files with your custom instructions that RagBot.AI should follow. The files int this folder contains the initial system instructions that set the context for the conversation. You can make a copy of the Rajiv's examples and modify those files to include any specific information or instructions you want to provide to RagBot.AI before starting the conversation.

To make a copy of Rajiv's own example custom instructions to modify for your own use, make a copy of the example folder for your own instructions and then edit the files using a text or markdown editor.

```bash
cp -rp examples/templates/instructions/starter instructions
```


2.  Set up your datasets for RagBot.AI.

Make a copy of Rajiv's sample files in the `examples/templates/datasets/starter/` folder to your own `datasets/` folder. This folder contains files that provide additional context and information to RagBot.AI. You can replace these sample files with your own information that reflect your personal preferences, such as your job details, family information, travel and food preferences, or any other information you want RagBot.AI to be aware of.

You can create new informational files or modify the existing ones to match your own needs. Each  file should contain relevant information related to a specific topic or aspect of your life. For example, you can create a `job-at-company-name.md` file to provide details about your work or a `hobbies.md` file to share information about your hobbies and interests.

Make sure to follow the Markdown format when creating or modifying these files, as RagBot.AI relies on Markdown syntax to parse and process the information.

```bash
cp -rp examples/templates/datasets/starter datasets
```

By personalizing the files in the `instructions/` and the `datasets/` folders with your own information, you can customize RagBot.AI to better understand your preferences and provide more accurate and relevant responses.

Remember to update the paths to your `instructions` and `datasets` folders and files in the `.env` configuration file to ensure that RagBot.AI uses the correct files during conversations.

Feel free to experiment and iterate on your personalized files to refine the context and information provided to RagBot.AI, making it an even more personalized AI assistant.

Now, RagBot.AI is configured, personalized, and ready to be run!