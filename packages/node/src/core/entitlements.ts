import { createHash } from "crypto";

export function fingerprintApiKey(apiKey?: string): string {
  const trimmed = apiKey?.trim();
  if (!trimmed) {
    return "";
  }
  return createHash("sha256").update(trimmed).digest("hex");
}
