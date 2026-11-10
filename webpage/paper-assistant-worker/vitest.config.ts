import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: {
        configPath: "./wrangler.mock.jsonc",
      },
    }),
  ],
  test: {
    coverage: {
      reporter: ["text", "html"],
    },
  },
});
