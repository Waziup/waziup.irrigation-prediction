import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, relative } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const entry = join(root, 'src/styles/base.css');
const failures = [];
const isImportOnly = (text) =>
  /^@import (['"])tailwindcss\1;$/.test(text.trim());
function inspect(directory) {
  for (const item of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    if (item.isDirectory()) {
      inspect(path);
    } else if (
      /\.(css|scss|sass|less|styl|stylus|pcss|postcss)$/i.test(item.name)
    ) {
      if (path !== entry || !isImportOnly(readFileSync(path, 'utf8')))
        failures.push(
          `${relative(root, path)}: use Tailwind utilities; custom styles are not permitted.`,
        );
    }
  }
}
inspect(join(root, 'src'));
inspect(join(root, 'public'));
const html = readFileSync(join(root, 'index.html'), 'utf8');
if (/<style\b|\bstyle\s*=/i.test(html))
  failures.push('index.html: inline CSS is not permitted.');
if (!isImportOnly(readFileSync(entry, 'utf8')))
  failures.push('CSS entry must contain only the Tailwind import.');
if (failures.length) {
  process.stderr.write(`${failures.join('\n')}\n`);
  process.exitCode = 1;
} else
  process.stdout.write(
    'Tailwind-only styling: CSS entry is import-only; no custom stylesheets.\n',
  );
