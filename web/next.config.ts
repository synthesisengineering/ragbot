import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Dockerfile's production stage runs the self-contained server that
  // `next build` writes to .next/standalone. Without this, that folder is never
  // written and the production image cannot build. `next dev` ignores it.
  output: "standalone",
};

export default nextConfig;
