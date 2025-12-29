import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";

import dotenv from "dotenv";
import { chromium } from "playwright";

type Provider = "openai" | "anthropic" | "xai" | "google";

interface CuratedModel {
  id: string;
  provider: Provider;
  displayName?: string;
  family?: string;
  capabilities?: Record<string, boolean>;
  contextWindow?: number;
  tokenPrices?: {
    input?: number;
    output?: number;
  };
  deprecated?: boolean;
  inPreview?: boolean;
}

interface DomSnapshot {
  title: string;
  headings: string[];
  tables: Array<{
    caption?: string;
    rows: string[][];
  }>;
  bodyText: string;
  nextData?: string;
  bodyHtml?: string;
}

interface ParsedPricing {
  provider: Provider;
  pricing: Array<{
    name: string;
    modelId?: string;
    inputPer1M?: number;
    outputPer1M?: number;
    contextWindow?: number;
    capabilities?: {
      text?: boolean;
      vision?: boolean;
      tool_use?: boolean;
      structured_output?: boolean;
      reasoning?: boolean;
    };
    notes?: string;
  }>;
}

interface ProviderSource {
  provider: Provider;
  pricingUrl: string;
}

const PROVIDER_SOURCES: ProviderSource[] = [
  {
    provider: "openai",
    pricingUrl: process.env.OPENAI_PRICING_URL ?? "https://openai.com/pricing",
  },
  {
    provider: "anthropic",
    pricingUrl:
      process.env.ANTHROPIC_PRICING_URL ??
      "https://www.anthropic.com/pricing",
  },
  {
    provider: "xai",
    pricingUrl: process.env.XAI_PRICING_URL ?? "https://docs.x.ai/docs/models",
  },
  {
    provider: "google",
    pricingUrl: process.env.GOOGLE_PRICING_URL ?? "https://ai.google.dev/pricing",
  },
];

const MAX_TEXT_CHARS = 60_000;
const MAX_NEXT_DATA_CHARS = 80_000;
const MAX_HTML_CHARS = 80_000;
const MIN_CONTENT_CHARS = Number(process.env.PRICING_MIN_CONTENT_CHARS ?? "200");
const PARSER_MODEL = process.env.PRICING_PARSER_MODEL ?? "gpt-4o-mini";
const VERBOSE = process.argv.includes("--verbose") || process.argv.includes("-v");
const NAV_TIMEOUT_MS = Number(process.env.PRICING_RENDER_TIMEOUT_MS ?? "90000");
const WAIT_FOR_SELECTOR = process.env.PRICING_WAIT_FOR_SELECTOR;
const LOG_BODY_PREVIEW_CHARS = Number(
  process.env.PRICING_LOG_BODY_PREVIEW_CHARS ?? "360",
);
const LOG_HEADINGS_LIMIT = Number(process.env.PRICING_LOG_HEADINGS_LIMIT ?? "8");
const LOG_TABLE_CAPTIONS_LIMIT = Number(
  process.env.PRICING_LOG_TABLE_CAPTIONS_LIMIT ?? "5",
);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(scriptDir, "..", ".env") });
dotenv.config({ path: path.join(scriptDir, "..", ".env.local") });

const curatedModelsPath = fileURLToPath(
  new URL("../../../models/curated_models.json", import.meta.url),
);

function normalizeWhitespace(input: string): string {
  return input.replace(/\s+/g, " ").trim();
}

function stripHtml(input: string): string {
  return input
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, "\"")
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}

const FORCE_COLOR = process.env.FORCE_COLOR && process.env.FORCE_COLOR !== "0";
const SUPPORTS_COLOR =
  FORCE_COLOR ||
  (process.stdout.isTTY && process.env.TERM !== "dumb" && !process.env.NO_COLOR);
const ANSI = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  gray: "\x1b[90m",
};
const ANSI_REGEX = /\x1b\[[0-9;]*m/g;
const LEVELS = {
  info: { label: "INFO", color: "cyan", write: console.log.bind(console) },
  warn: { label: "WARN", color: "yellow", write: console.warn.bind(console) },
  error: { label: "ERROR", color: "red", write: console.error.bind(console) },
  success: { label: "OK", color: "green", write: console.log.bind(console) },
  verbose: { label: "VERBOSE", color: "magenta", write: console.log.bind(console) },
} as const;

function colorize(text: string, color?: keyof typeof ANSI): string {
  if (!SUPPORTS_COLOR || !color || !ANSI[color]) {
    return text;
  }
  return `${ANSI[color]}${text}${ANSI.reset}`;
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms)) {
    return String(ms);
  }
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatValue(key: string, value: unknown): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value === null) {
    return "null";
  }
  if (typeof value === "number") {
    if (key === "ms") {
      return formatDuration(value);
    }
    if (key.endsWith("Chars")) {
      return `${value} chars`;
    }
    return `${value}`;
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    if (value.every((item) => typeof item === "string" || typeof item === "number")) {
      return value.join(", ");
    }
  }
  return inspect(value, {
    colors: false,
    depth: 4,
    compact: true,
    breakLength: 100,
  });
}

function formatDetails(details?: Record<string, unknown>): string[] {
  if (!details) {
    return [];
  }
  const entries = Object.entries(details).filter(([, value]) => value !== undefined);
  if (!entries.length) {
    return [];
  }
  const keyWidth = Math.min(
    Math.max(...entries.map(([key]) => key.length)),
    22,
  );
  return entries.flatMap(([key, value]) => {
    const formatted = formatValue(key, value);
    if (formatted === undefined) {
      return [];
    }
    return String(formatted)
      .split("\n")
      .map((line, index) => {
        const keyLabel = index === 0 ? key.padEnd(keyWidth) : " ".repeat(keyWidth);
        const dimKey = colorize(keyLabel, "dim");
        return `${dimKey} : ${line}`;
      });
  });
}

function previewText(text: string | undefined, maxChars: number): string | undefined {
  if (!text || maxChars <= 0) {
    return undefined;
  }
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, maxChars)}...`;
}

function limitList(list: string[] | undefined, maxItems: number): string[] | undefined {
  if (!list || list.length === 0 || maxItems <= 0) {
    return undefined;
  }
  if (list.length <= maxItems) {
    return list;
  }
  return [...list.slice(0, maxItems), `+${list.length - maxItems} more`];
}

function formatMatchedEntry(
  record: ParsedPricing["pricing"][number],
  match: CuratedModel,
): string {
  const input = record.inputPer1M ?? "-";
  const output = record.outputPer1M ?? "-";
  const context = record.contextWindow ?? "-";
  return `${record.name} -> ${match.id} (input=${input}, output=${output}, ctx=${context})`;
}

function logWithDetails(
  levelKey: keyof typeof LEVELS,
  message: string,
  details?: Record<string, unknown>,
) {
  const level = LEVELS[levelKey];
  const labelText = `[${level.label}]`;
  const label = colorize(labelText, level.color);
  const line = `${label} ${message}`;
  level.write(line);
  const detailLines = formatDetails(details);
  if (!detailLines.length) {
    return;
  }
  const indent = " ".repeat(labelText.length + 1);
  for (const detailLine of detailLines) {
    const lineText = SUPPORTS_COLOR
      ? detailLine
      : detailLine.replace(ANSI_REGEX, "");
    level.write(`${indent}${lineText}`);
  }
}

function logInfo(message: string, details?: Record<string, unknown>) {
  logWithDetails("info", message, details);
}

function logWarn(message: string, details?: Record<string, unknown>) {
  logWithDetails("warn", message, details);
}

function logError(message: string, details?: Record<string, unknown>) {
  logWithDetails("error", message, details);
}

function logSuccess(message: string, details?: Record<string, unknown>) {
  logWithDetails("success", message, details);
}

function logVerbose(message: string, details?: Record<string, unknown>) {
  if (!VERBOSE) {
    return;
  }
  logWithDetails("verbose", message, details);
}

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  logError("Missing OPENAI_API_KEY for pricing parser.");
  process.exit(1);
}

function normalizeName(input?: string): string {
  if (!input) {
    return "";
  }
  return normalizeWhitespace(input).toLowerCase();
}

async function loadCuratedModels(): Promise<CuratedModel[]> {
  const raw = await fs.readFile(curatedModelsPath, "utf-8");
  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? (parsed as CuratedModel[]) : [];
}

async function saveCuratedModels(models: CuratedModel[]): Promise<void> {
  const next = JSON.stringify(models, null, 2) + "\n";
  await fs.writeFile(curatedModelsPath, next, "utf-8");
}

function providerWaitSelector(provider: Provider): string | undefined {
  switch (provider) {
    case "openai":
      return process.env.OPENAI_WAIT_FOR_SELECTOR ?? WAIT_FOR_SELECTOR;
    case "anthropic":
      return process.env.ANTHROPIC_WAIT_FOR_SELECTOR ?? WAIT_FOR_SELECTOR;
    case "xai":
      return process.env.XAI_WAIT_FOR_SELECTOR ?? WAIT_FOR_SELECTOR;
    case "google":
      return process.env.GOOGLE_WAIT_FOR_SELECTOR ?? WAIT_FOR_SELECTOR;
    default:
      return WAIT_FOR_SELECTOR;
  }
}

async function captureDomSnapshot(
  url: string,
  waitForSelector?: string,
): Promise<DomSnapshot> {
  const start = Date.now();
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
  });
  const page = await context.newPage();
  let response: Awaited<ReturnType<typeof page.goto>> | null = null;
  try {
    response = await page.goto(url, {
      waitUntil: "networkidle",
      timeout: NAV_TIMEOUT_MS,
    });
  } catch (error) {
    if ((error as { name?: string }).name !== "TimeoutError") {
      throw error;
    }
    logVerbose("Network idle timeout; retrying with domcontentloaded", { url });
    response = await page.goto(url, {
      waitUntil: "domcontentloaded",
      timeout: NAV_TIMEOUT_MS,
    });
  }
  if (WAIT_FOR_SELECTOR) {
    logVerbose("Waiting for selector", { url, selector: WAIT_FOR_SELECTOR });
    await page.waitForSelector(WAIT_FOR_SELECTOR, { timeout: NAV_TIMEOUT_MS });
  }
  if (waitForSelector && waitForSelector !== WAIT_FOR_SELECTOR) {
    logVerbose("Waiting for provider selector", {
      url,
      selector: waitForSelector,
    });
    await page.waitForSelector(waitForSelector, { timeout: NAV_TIMEOUT_MS });
  }
  await page.waitForTimeout(1500);
  const snapshot = await page.evaluate(() => {
    const text = document.body?.innerText ?? "";
    const headings = Array.from(
      document.querySelectorAll("h1, h2, h3, h4"),
    )
      .map((node) => node.textContent?.trim() ?? "")
      .filter(Boolean);
    const tables = Array.from(document.querySelectorAll("table")).map(
      (table) => {
        const caption = table.querySelector("caption")?.textContent?.trim();
        const rows = Array.from(table.querySelectorAll("tr")).map((row) =>
          Array.from(row.querySelectorAll("th, td")).map((cell) =>
            (cell.textContent ?? "").trim(),
          ),
        );
        return { caption, rows };
      },
    );
    const nextData =
      document.querySelector("#__NEXT_DATA__")?.textContent ?? "";
    const bodyHtml = document.body?.innerHTML ?? "";
    return {
      title: document.title,
      headings,
      tables,
      bodyText: text,
      nextData,
      bodyHtml,
    };
  });
  let bodyText = normalizeWhitespace(snapshot.bodyText);
  if (!bodyText || bodyText.length < MIN_CONTENT_CHARS) {
    const html = await page.content().catch(() => "");
    if (html) {
      bodyText = stripHtml(html);
    }
  }
  await page.close();
  await context.close();
  await browser.close();
  const normalized = {
    ...snapshot,
    bodyText: bodyText.slice(0, MAX_TEXT_CHARS),
    nextData: snapshot.nextData
      ? snapshot.nextData.slice(0, MAX_NEXT_DATA_CHARS)
      : undefined,
    bodyHtml:
      bodyText.length < MIN_CONTENT_CHARS
        ? snapshot.bodyHtml?.slice(0, MAX_HTML_CHARS)
        : undefined,
  };
  logVerbose("Rendered pricing page", {
    url,
    ms: Date.now() - start,
    status: response?.status(),
    headings: normalized.headings.length,
    tables: normalized.tables.length,
    bodyTextChars: normalized.bodyText.length,
    nextDataChars: normalized.nextData?.length ?? 0,
    bodyHtmlChars: normalized.bodyHtml?.length ?? 0,
  });
  const tableCaptions = normalized.tables.map((table, index) => {
    const caption = table.caption?.trim();
    if (caption) {
      return caption;
    }
    return `Table ${index + 1} (${table.rows.length} rows)`;
  });
  logVerbose("Snapshot excerpt", {
    url,
    title: normalized.title,
    headings: limitList(normalized.headings, LOG_HEADINGS_LIMIT),
    tableCaptions: limitList(tableCaptions, LOG_TABLE_CAPTIONS_LIMIT),
    bodyTextPreview: previewText(normalized.bodyText, LOG_BODY_PREVIEW_CHARS),
  });
  return normalized;
}

function buildPrompt(provider: Provider, url: string, snapshot: DomSnapshot): string {
  return [
    `Provider: ${provider}`,
    `Source URL: ${url}`,
    "",
    "Extract explicit model pricing and capability data from the DOM snapshot.",
    "Rules:",
    "- Use only explicit data in the snapshot. Do not guess or infer.",
    "- Use per-1M token prices when stated; if only per-1K is stated, convert to per-1M.",
    "- If a model lacks explicit input/output prices, omit those fields.",
    "- If neither input nor output pricing is explicit, omit the model.",
    "- If capabilities or context window are explicitly stated, include them.",
    "- Use the exact model display name as it appears in the snapshot.",
    "",
    "DOM snapshot follows as JSON (may include nextData JSON if present):",
    JSON.stringify(snapshot),
  ].join("\n");
}

async function parsePricing(
  provider: Provider,
  url: string,
  snapshot: DomSnapshot,
): Promise<ParsedPricing> {
  logVerbose("Parsing pricing via OpenAI", {
    provider,
    model: PARSER_MODEL,
    url,
    snapshotChars: snapshot.bodyText.length,
    tableCount: snapshot.tables.length,
    nextDataChars: snapshot.nextData?.length ?? 0,
    bodyHtmlChars: snapshot.bodyHtml?.length ?? 0,
  });
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: PARSER_MODEL,
      input: [
        {
          role: "system",
          content:
            "You extract structured data from HTML snapshots. Never invent values.",
        },
        {
          role: "user",
          content: buildPrompt(provider, url, snapshot),
        },
      ],
      temperature: 0,
      max_output_tokens: 2000,
      text: {
        format: {
          type: "json_schema",
          name: "pricing_parse",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["provider", "pricing"],
            properties: {
              provider: { type: "string" },
              pricing: {
                type: "array",
                items: {
                  type: "object",
                  additionalProperties: false,
                  required: [
                    "name",
                    "modelId",
                    "inputPer1M",
                    "outputPer1M",
                    "contextWindow",
                    "capabilities",
                    "notes",
                  ],
                  properties: {
                    name: { type: "string" },
                    modelId: { type: ["string", "null"] },
                    inputPer1M: { type: ["number", "null"] },
                    outputPer1M: { type: ["number", "null"] },
                    contextWindow: { type: ["integer", "null"] },
                    capabilities: {
                      type: ["object", "null"],
                      additionalProperties: false,
                      required: [
                        "text",
                        "vision",
                        "tool_use",
                        "structured_output",
                        "reasoning",
                      ],
                      properties: {
                        text: { type: ["boolean", "null"] },
                        vision: { type: ["boolean", "null"] },
                        tool_use: { type: ["boolean", "null"] },
                        structured_output: { type: ["boolean", "null"] },
                        reasoning: { type: ["boolean", "null"] },
                      },
                    },
                    notes: { type: ["string", "null"] },
                  },
                },
              },
            },
          },
        },
      },
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `OpenAI parser failed (${response.status}): ${errorBody || "unknown error"}`,
    );
  }

  const requestId =
    response.headers.get("x-request-id") ??
    response.headers.get("x-request-id".toLowerCase());
  const data = await response.json();
  const usage = data.usage ?? data.response?.usage;
  if (usage) {
    logVerbose("OpenAI usage", { provider, requestId, usage });
  } else {
    logVerbose("OpenAI response received", { provider, requestId });
  }
  const outputText =
    data.output_text ??
    data.output
      ?.flatMap(
        (entry: { content?: Array<{ type: string; text?: string }> }) =>
          entry.content ?? [],
      )
      .find((item: { type: string; text?: string }) => item.type === "output_text")
      ?.text ??
    data.output?.[0]?.content?.[0]?.text;
  if (!outputText) {
    throw new Error("OpenAI parser returned no output text.");
  }
  return JSON.parse(outputText) as ParsedPricing;
}

function findCuratedMatch(
  models: CuratedModel[],
  provider: Provider,
  record: ParsedPricing["pricing"][number],
): CuratedModel | undefined {
  if (record.modelId) {
    const direct = models.find(
      (model) =>
        model.provider === provider && normalizeName(model.id) === normalizeName(record.modelId),
    );
    if (direct) {
      return direct;
    }
  }
  const name = normalizeName(record.name);
  if (!name) {
    return undefined;
  }
  const matches = models.filter(
    (model) =>
      model.provider === provider &&
      normalizeName(model.displayName ?? model.id) === name,
  );
  if (matches.length === 1) {
    return matches[0];
  }
  return undefined;
}

function applyParsedRecord(
  model: CuratedModel,
  record: ParsedPricing["pricing"][number],
): CuratedModel {
  const tokenPrices = { ...(model.tokenPrices ?? {}) };
  if (record.inputPer1M !== undefined && record.inputPer1M !== null) {
    tokenPrices.input = record.inputPer1M;
  }
  if (record.outputPer1M !== undefined && record.outputPer1M !== null) {
    tokenPrices.output = record.outputPer1M;
  }
  const capabilities = record.capabilities
    ? { ...(model.capabilities ?? {}), ...record.capabilities }
    : model.capabilities;
  return {
    ...model,
    tokenPrices:
      (record.inputPer1M !== undefined && record.inputPer1M !== null) ||
      (record.outputPer1M !== undefined && record.outputPer1M !== null)
        ? tokenPrices
        : model.tokenPrices,
    contextWindow:
      record.contextWindow !== undefined && record.contextWindow !== null
        ? record.contextWindow
        : model.contextWindow,
    capabilities,
  };
}

async function main() {
  const curatedModels = await loadCuratedModels();
  const updatedModels = [...curatedModels];
  const missingMatches: string[] = [];
  const matchedEntriesByProvider: Record<Provider, string[]> = {
    openai: [],
    anthropic: [],
    xai: [],
    google: [],
  };
  let isFirstProvider = true;

  for (const source of PROVIDER_SOURCES) {
    if (!isFirstProvider) {
      console.log("");
    }
    isFirstProvider = false;
    logInfo(`Scraping ${source.provider} pricing`, { url: source.pricingUrl });
    const waitForSelector = providerWaitSelector(source.provider);
    const snapshot = await captureDomSnapshot(source.pricingUrl, waitForSelector);
    if (
      snapshot.bodyText.length < MIN_CONTENT_CHARS &&
      !snapshot.tables.length &&
      !(snapshot.nextData && snapshot.nextData.length) &&
      !(snapshot.bodyHtml && snapshot.bodyHtml.length)
    ) {
      throw new Error(
        `No DOM content captured for ${source.pricingUrl}. Try OPENAI_WAIT_FOR_SELECTOR (or provider-specific selector) or increase PRICING_RENDER_TIMEOUT_MS.`,
      );
    }
    const parsed = await parsePricing(source.provider, source.pricingUrl, snapshot);
    logVerbose("Parsed pricing entries", {
      provider: source.provider,
      entries: parsed.pricing.length,
    });
    for (const record of parsed.pricing) {
      const match = findCuratedMatch(updatedModels, source.provider, record);
      if (!match) {
        missingMatches.push(`${source.provider}:${record.name}`);
        continue;
      }
      matchedEntriesByProvider[source.provider].push(
        formatMatchedEntry(record, match),
      );
      const index = updatedModels.findIndex(
        (model) => model.provider === match.provider && model.id === match.id,
      );
      if (index >= 0) {
        updatedModels[index] = applyParsedRecord(updatedModels[index], record);
      }
    }
    const matchedEntries = matchedEntriesByProvider[source.provider];
    if (matchedEntries.length) {
      logVerbose("Matched pricing entries", {
        provider: source.provider,
        entries: matchedEntries.join("\n"),
      });
    } else {
      logVerbose("Matched pricing entries", {
        provider: source.provider,
        entries: "none",
      });
    }
  }

  if (missingMatches.length) {
    logWarn("No curated match for entries", {
      count: missingMatches.length,
      entries: missingMatches,
    });
  }
  const totalMatched = Object.values(matchedEntriesByProvider).reduce(
    (sum, entries) => sum + entries.length,
    0,
  );
  logInfo("Matched pricing entries", { count: totalMatched });

  await saveCuratedModels(updatedModels);
  logSuccess("Updated curated models", { count: updatedModels.length });
}

main().catch((error) => {
  if (error instanceof Error) {
    logError("Refresh failed", { error: error.message });
    if (VERBOSE && error.stack) {
      console.error(colorize(error.stack, "dim"));
    }
  } else {
    logError("Refresh failed", { error });
  }
  process.exit(1);
});
