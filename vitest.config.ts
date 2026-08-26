import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['tests/**/*.spec.ts'],
    coverage: {
      include: ['src/**'],
      // Types-only files have no runtime coverage (the monorepo gate excludes
      // src/types.ts the same way).
      exclude: ['src/types.ts'],
      reporter: ['text', 'json-summary'],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      },
    },
  },
})
