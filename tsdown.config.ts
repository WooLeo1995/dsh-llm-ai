import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['src/index.ts', 'src/invariant.ts'],
  outDir: 'lib',
  format: 'esm',
  dts: true,
  entryFileNames: '[name].js',
  clean: true,
})
