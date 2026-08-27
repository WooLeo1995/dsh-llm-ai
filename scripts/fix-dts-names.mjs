// tsdown's dts plugin emits hashed declaration names (index-AbCdEf.d.ts) even
// with entryFileNames pinned; the exports map promises unhashed lib/*.d.ts.
// Entries have unique base names, so stripping the hash is collision-free.
import { readdirSync, renameSync } from 'node:fs'
import { join } from 'node:path'

const dir = new URL('../lib', import.meta.url).pathname
for (const file of readdirSync(dir)) {
  const match = /^(?<name>[A-Za-z0-9._-]+?)-[A-Za-z0-9_-]{8}\.d\.ts$/.exec(file)
  if (match?.groups?.name !== undefined) {
    renameSync(join(dir, file), join(dir, `${match.groups.name}.d.ts`))
    console.log(`renamed ${file} -> ${match.groups.name}.d.ts`)
  }
}
