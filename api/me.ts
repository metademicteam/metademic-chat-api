import type { IncomingMessage, ServerResponse } from "node:http";
import { applyCors, env } from "./_shared.js";

/**
 * Role probe used by the site's chat page. This compatibility deployment
 * resolves the caller's Supabase session exactly like the site's own
 * /api/me so stale cached bundles keep working cross-origin.
 */
export default async function handler(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  if (!applyCors(req, res)) {
    res.statusCode = 403;
    res.end("Origin not allowed");
    return;
  }
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }
  if (req.method !== "GET") {
    res.statusCode = 405;
    res.end("Use GET for /api/me.");
    return;
  }

  const header = req.headers.authorization ?? "";
  const token =
    (typeof header === "string" ? header.replace(/^Bearer\s+/i, "") : "") ||
    (typeof req.headers["x-supabase-token"] === "string"
      ? req.headers["x-supabase-token"]
      : "");

  const base = env("SUPABASE_URL").replace(/\/$/, "");
  const anon = env("SUPABASE_ANON_KEY");
  const serviceKey = env("SUPABASE_SERVICE_ROLE_KEY");

  let role = "visitor";
  let name: string | null = null;
  if (base && token) {
    try {
      const userResponse = await fetch(`${base}/auth/v1/user`, {
        headers: {
          apikey: anon || serviceKey,
          Authorization: `Bearer ${token}`,
        },
      });
      if (userResponse.ok) {
        const user = (await userResponse.json()) as {
          id?: string;
          email?: string;
        };
        if (user?.id) {
          role = "member";
          name = user.email ? user.email.split("@")[0] : null;
          if (serviceKey) {
            const profileResponse = await fetch(
              `${base}/rest/v1/profiles?user_id=eq.${encodeURIComponent(user.id)}&select=role,display_name`,
              {
                headers: {
                  apikey: serviceKey,
                  Authorization: `Bearer ${serviceKey}`,
                },
              },
            );
            if (profileResponse.ok) {
              const rows = (await profileResponse.json()) as {
                role?: string;
                display_name?: string | null;
              }[];
              if (rows[0]?.role === "admin" || rows[0]?.role === "member") {
                role = rows[0].role;
              }
              if (rows[0]?.display_name) name = rows[0].display_name;
            }
          }
        }
      }
    } catch {
      // fall through as visitor
    }
  }

  const slackRelayAvailable = Boolean(
    env("SLACK_WEBHOOK_URL") ||
      (env("SLACK_BOT_TOKEN") && env("SLACK_CHANNEL_ID")),
  );

  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ role, name, slackRelayAvailable }));
}
