import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import {
	fauxAssistantMessage,
	fauxToolCall,
	getCurrentSystemMessage,
	getCurrentSystemPrompt,
	getSystemMessageText,
	type Model,
	resolveTranscriptTools,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { getModel, streamSimple } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { describe, expect, test } from "vitest";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import {
	buildSystemPromptSections,
	buildSystemPromptState,
	diffSystemPromptSections,
} from "../src/core/system-prompt.ts";
import { createToolSearchExtension } from "../src/extensions/tool-search/index.ts";
import type { ExtensionFactory } from "../src/index.ts";
import { createHarness, type Harness } from "./suite/harness.ts";

describe("system prompt updates", () => {
	test("declares the prompt and tools once and reuses them across resume", async () => {
		const harness = await createHarness();
		try {
			harness.setResponses([fauxAssistantMessage("first"), fauxAssistantMessage("second")]);
			await harness.session.prompt("one");
			await harness.session.prompt("two");
			const systemEntries = harness.sessionManager
				.getEntries()
				.filter((entry) => entry.type === "message" && entry.message.role === "system");
			expect(systemEntries).toHaveLength(1);
			expect(harness.session.messages.map((message) => message.role)).toEqual([
				"system",
				"user",
				"assistant",
				"user",
				"assistant",
			]);
			const head = harness.session.messages[0];
			if (head?.role !== "system") throw new Error("expected system message");
			expect(head.content).toBe("");
			expect(Object.keys(head.sections ?? {})).toEqual(["preamble", "tools", "rules", "docs", "cwd"]);
			expect(head.toolsAdded?.map((tool) => tool.name)).toEqual(["read", "bash", "edit", "write"]);
			expect(getSystemMessageText(head)).toBe(harness.session.systemPrompt);
		} finally {
			harness.cleanup();
		}
	});

	// #11495
	test("does not patch the prompt when the active tools are only reordered", async () => {
		const harness = await createHarness();
		try {
			harness.setResponses([fauxAssistantMessage("first"), fauxAssistantMessage("second")]);
			await harness.session.prompt("one");
			const before = harness.session.getActiveToolNames();
			harness.session.setActiveToolsByName([...before].reverse());
			expect(harness.session.getActiveToolNames()).toEqual(before);
			await harness.session.prompt("two");
			const systemMessages = harness.session.messages.filter((message) => message.role === "system");
			expect(systemMessages).toHaveLength(1);
		} finally {
			harness.cleanup();
		}
	});

	test("opens a transcript without a system message and declares the prompt on the first request", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-system-prompt-migration-"));
		try {
			const sessionManager = SessionManager.inMemory(tempDir);
			sessionManager.appendMessage({ role: "user", content: "existing", timestamp: 1 });
			const created = await createAgentSession({
				cwd: tempDir,
				agentDir: join(tempDir, "agent"),
				model: getModel("anthropic", "claude-sonnet-4-5")!,
				settingsManager: SettingsManager.inMemory(),
				sessionManager,
				noTools: "all",
			});
			try {
				// Nothing is synthesized or persisted until a request needs it.
				expect(created.session.messages.map((message) => message.role)).toEqual(["user"]);
				expect(sessionManager.buildSessionContext().messages.map((message) => message.role)).toEqual(["user"]);
				expect(getCurrentSystemMessage(created.session.messages)).toBeUndefined();
			} finally {
				created.session.dispose();
			}
		} finally {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	test("diffs sections into a patch", () => {
		const previous = buildSystemPromptSections({ cwd: "/tmp", sections: { plan_mode: "Plan only." } });
		const current = buildSystemPromptSections({ cwd: "/tmp", sections: { plan_mode: "Implementation allowed." } });
		expect(diffSystemPromptSections(previous, current)).toEqual({
			plan_mode: "<plan_mode>\nImplementation allowed.\n</plan_mode>",
		});
		expect(diffSystemPromptSections(previous, previous)).toBeUndefined();
		expect(diffSystemPromptSections(previous, buildSystemPromptSections({ cwd: "/tmp" }))).toEqual({
			plan_mode: null,
		});
	});

	test("keeps the preamble untagged and replaces it like any section", () => {
		const previous = buildSystemPromptSections({ customPrompt: "You are A.", cwd: "/tmp" });
		const current = buildSystemPromptSections({ customPrompt: "You are B.", cwd: "/tmp" });
		expect(previous.preamble).toBe("You are A.");
		expect(diffSystemPromptSections(previous, current)).toEqual({ preamble: "You are B." });

		expect(buildSystemPromptState({ forceSystemPrompt: "Exact prompt.", cwd: "/tmp" })).toEqual({
			content: "Exact prompt.",
		});
		expect(buildSystemPromptState({ cwd: "/tmp" })).toEqual({
			content: "",
			sections: buildSystemPromptSections({ cwd: "/tmp" }),
		});
		expect(() => buildSystemPromptSections({ cwd: "/tmp", sections: { preamble: "x" } })).toThrow(
			"Invalid system prompt section name",
		);
	});

	test("a forced prompt is sent as the leading prompt for the run and never recorded", async () => {
		let turn = 0;
		const extension: ExtensionFactory = (pi) => {
			pi.on("before_agent_start", (event) => {
				if (++turn === 3) event.systemPromptOptions.sections.plan_mode = "Plan only.";
				return turn === 2 || turn === 3 ? { systemPrompt: "Exact prompt." } : undefined;
			});
		};
		const harness = await createHarness({ extensionFactories: [extension] });
		try {
			const requests: TranscriptContext[] = [];
			harness.setResponses(
				["one", "two", "three", "four"].map((text) => (providerContext: TranscriptContext) => {
					requests.push(providerContext);
					return fauxAssistantMessage(text);
				}),
			);
			for (const text of ["one", "two", "three", "four"]) await harness.session.prompt(text);
			const systemMessages = requests.map((request) =>
				request.messages.filter((message) => message.role === "system"),
			);
			// Forced turns collapse to one leading message; the unforced fourth turn passes the
			// recorded head and both plan_mode patches through.
			expect(systemMessages.map((messages) => messages.length)).toEqual([1, 1, 1, 3]);

			const forced = systemMessages[1]?.at(-1);
			expect(forced).toEqual({
				role: "system",
				content: "Exact prompt.",
				toolsAdded: systemMessages[0]?.[0]?.toolsAdded,
				timestamp: systemMessages[0]?.[0]?.timestamp,
			});
			expect(systemMessages[2]?.at(-1)).toEqual(forced);
			expect(getCurrentSystemPrompt(requests[2]!.messages)).toBe("Exact prompt.");
			expect(requests[2]!.messages.map((message) => message.role)).toEqual([
				"system",
				"user",
				"assistant",
				"user",
				"assistant",
				"user",
			]);

			// The transcript only records the structured sections, never the forced text.
			const recorded = harness.session.messages.flatMap((message) =>
				message.role === "system" ? [message.sections] : [],
			);
			expect(recorded).toEqual([
				systemMessages[0]?.[0]?.sections,
				{ plan_mode: "<plan_mode>\nPlan only.\n</plan_mode>" },
				{ plan_mode: null },
			]);
			expect(getCurrentSystemPrompt(harness.session.messages)).toBe(harness.session.systemPrompt);
		} finally {
			harness.cleanup();
		}
	});

	// #10489: the forced prompt head keeps the initial tool declarations, so a tool_search
	// discovery stays on its own later system message instead of growing the request tool
	// list and re-billing the cached prefix on every discovery.
	const forcedPromptFactories = (mode: "systemPrompt" | "forceSystemPrompt") => [
		createToolSearchExtension(),
		((pi) => {
			pi.registerTool({
				name: "search_docs",
				label: "Search docs",
				description: "Search the docs.",
				exposure: "deferred",
				parameters: Type.Object({}),
				execute: async () => ({ content: [], details: {} }),
			});
			pi.on("before_agent_start", (event) => {
				if (mode === "systemPrompt") return { systemPrompt: "Exact prompt." };
				event.systemPromptOptions.forceSystemPrompt = "Exact prompt.";
			});
		}) satisfies ExtensionFactory,
	];

	/** Loads search_docs through a real tool_search call, then withdraws it: three requests. */
	async function runForcedToolSearchTurns(harness: Harness): Promise<TranscriptContext[]> {
		const requests: TranscriptContext[] = [];
		harness.setResponses([
			(context) => {
				requests.push(context);
				return fauxAssistantMessage(fauxToolCall("tool_search", { query: "search docs", limit: 1 }), {
					stopReason: "toolUse",
				});
			},
			(context) => {
				requests.push(context);
				return fauxAssistantMessage("loaded");
			},
			(context) => {
				requests.push(context);
				return fauxAssistantMessage("removed");
			},
		]);
		await harness.session.prompt("find docs");
		// The tool_search call and its result take one extra request round trip.
		expect(requests).toHaveLength(2);
		harness.session.setActiveToolsByName(["tool_search"]);
		await harness.session.prompt("continue");
		expect(requests).toHaveLength(3);
		return requests;
	}

	test.each(["systemPrompt", "forceSystemPrompt"] as const)(
		"%s keeps tool_search additions out of the forced prompt head",
		async (mode) => {
			const harness = await createHarness({
				initialActiveToolNames: ["tool_search"],
				extensionFactories: forcedPromptFactories(mode),
			});
			// The faux provider never opens a connection; fail loudly if any transport is attempted.
			const originalFetch = globalThis.fetch;
			let fetchAttempts = 0;
			globalThis.fetch = (async () => {
				fetchAttempts++;
				throw new Error("network transport attempted in an offline test");
			}) as typeof fetch;
			try {
				const requests = await runForcedToolSearchTurns(harness);
				const head = requests[0]?.messages[0];
				if (head?.role !== "system") throw new Error("expected a leading system message");
				expect(head.content).toBe("Exact prompt.");
				expect(head.toolsAdded?.map((tool) => tool.name)).toEqual(["tool_search"]);

				// The request tool list keeps the initial declarations byte-for-byte.
				const initial = resolveTranscriptTools(requests[0]!.messages, "additions");
				const loaded = resolveTranscriptTools(requests[1]!.messages, "additions");
				expect(loaded.requestTools).toEqual(initial.requestTools);
				// The discovered tool stays anchored to its own later system message.
				const delta = requests[1]!.messages.filter((message) => message.role === "system").at(-1);
				expect(delta?.toolsAdded?.map((tool) => tool.name)).toEqual(["search_docs"]);
				expect(delta?.content).toBe("");
				expect(delta?.sections).toBeUndefined();
				expect(requests[1]!.messages.indexOf(delta!)).toBeGreaterThan(
					requests[1]!.messages.findIndex((message) => message.role === "toolResult"),
				);
				// The forced text is the effective prompt; the transcript keeps its structured sections.
				expect(getCurrentSystemPrompt(requests[1]!.messages)).toBe("Exact prompt.");
				expect(getCurrentSystemPrompt(harness.session.messages)).not.toBe("Exact prompt.");

				// The removal survives the projection, so the withdrawn tool is not re-declared.
				const removed = resolveTranscriptTools(requests[2]!.messages, "additions");
				expect(removed.requestTools.map((tool) => tool.name)).toEqual(["tool_search"]);
				const removal = requests[2]!.messages.filter((message) => message.role === "system").at(-1);
				expect(removal?.toolsRemoved).toEqual([{ name: "search_docs" }]);
				expect(fetchAttempts).toBe(0);
			} finally {
				globalThis.fetch = originalFetch;
				harness.cleanup();
			}
		},
	);

	// #10489: wire-level regression. The projection is only useful if the serialized request
	// keeps its cached prefix, so replay the captured requests through the Responses transport
	// that declares tools in additional_tools items.
	const additionalToolsModel: Model<"openai-responses"> = {
		id: "gpt-responses-test",
		name: "GPT Responses Test",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "http://127.0.0.1:9",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 100000,
		maxTokens: 1000,
		compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
	};

	interface ResponsesPayload {
		tools?: Array<{ name: string }>;
		input: Array<{
			type?: string;
			role?: string;
			content?: string;
			tools?: Array<{ name: string }>;
		}>;
	}

	/** Serialize a request like the Responses transport would and capture it before any network. */
	async function captureResponsesPayload(context: TranscriptContext): Promise<ResponsesPayload> {
		let captured: ResponsesPayload | undefined;
		const stream = streamSimple(additionalToolsModel, context, {
			apiKey: "test-key",
			onPayload: (payload) => {
				captured = payload as ResponsesPayload;
				throw new Error("payload captured; tests never open a transport");
			},
		});
		await stream.result();
		if (!captured) throw new Error("expected payload capture");
		return captured;
	}

	test("a forced prompt keeps the serialized Responses prefix across tool_search changes", async () => {
		const harness = await createHarness({
			initialActiveToolNames: ["tool_search"],
			extensionFactories: forcedPromptFactories("forceSystemPrompt"),
		});
		try {
			const requests = await runForcedToolSearchTurns(harness);
			const payloads = [];
			for (const request of requests) {
				payloads.push(await captureResponsesPayload(request));
			}
			const [initial, loaded, removed] = payloads;

			// The leading declaration and the forced instructions are the cached prefix.
			expect(initial.tools).toBeUndefined();
			expect(initial.input[0]).toMatchObject({
				type: "additional_tools",
				tools: [expect.objectContaining({ name: "tool_search" })],
			});
			expect(initial.input[1]).toMatchObject({ role: "developer", content: "Exact prompt." });

			// Discovery appends an additional_tools item; earlier items are byte-identical.
			expect(loaded.input.slice(0, initial.input.length)).toEqual(initial.input);
			expect(
				loaded.input
					.filter((item) => item.type === "additional_tools")
					.map((item) => item.tools?.map((tool) => tool.name)),
			).toEqual([["tool_search"], ["search_docs"]]);

			// Removal appends its announcement; nothing earlier is rewritten and no removed
			// tool is declared again.
			expect(removed.input.slice(0, loaded.input.length)).toEqual(loaded.input);
			const announcement = removed.input.find((item) => item.content?.includes("no longer available"));
			expect(announcement?.content).toContain("- search_docs");
			expect(
				removed.input
					.filter((item) => item.type === "additional_tools")
					.map((item) => item.tools?.map((tool) => tool.name)),
			).toEqual([["tool_search"], ["search_docs"]]);
			expect(removed.tools).toBeUndefined();
		} finally {
			harness.cleanup();
		}
	});

	test("setActiveTools emits prompt sections and tool changes before the next request", async () => {
		const extension: ExtensionFactory = (pi) => {
			for (const name of ["first", "second"]) {
				pi.registerTool({
					name,
					label: name,
					description: `${name} description`,
					promptSnippet: `${name} prompt snippet`,
					promptGuidelines: [`Use ${name} carefully.`],
					parameters: Type.Object({}),
					execute: async () => ({ content: [{ type: "text", text: name }], details: {} }),
				});
			}
		};
		const harness = await createHarness({ extensionFactories: [extension], initialActiveToolNames: ["first"] });
		try {
			// Faux response callbacks swallow thrown assertions, so capture and assert afterwards.
			const requests: TranscriptContext[] = [];
			harness.setResponses([
				(providerContext) => {
					requests.push(providerContext);
					return fauxAssistantMessage("first");
				},
				(providerContext) => {
					requests.push(providerContext);
					return fauxAssistantMessage([fauxToolCall("first", {})], { stopReason: "toolUse" });
				},
				(providerContext) => {
					requests.push(providerContext);
					return fauxAssistantMessage("second");
				},
			]);
			await harness.session.prompt("first");
			harness.session.setActiveToolsByName(["second"]);
			await harness.session.prompt("second");
			expect(requests).toHaveLength(3);

			expect(Object.keys(requests[0] ?? {})).toEqual(["messages"]);
			const initial = requests[0]?.messages[0];
			if (initial?.role !== "system") throw new Error("expected initial system message");
			expect(initial.toolsAdded?.map((value) => value.name)).toEqual(["first", "second"]);
			expect(initial.sections?.tools).toContain("first prompt snippet");

			const update = requests[1]?.messages.filter((message) => message.role === "system").at(-1);
			expect(update).toEqual({
				role: "system",
				content: "",
				sections: { tools: expect.stringContaining("second prompt snippet"), rules: expect.any(String) },
				toolsRemoved: [{ name: "first" }],
				timestamp: expect.any(Number),
			});
			expect(update?.sections?.tools).not.toContain("first prompt snippet");
			expect(update?.sections?.rules).not.toContain("Use first carefully.");

			const result = requests[2]?.messages.filter((message) => message.role === "toolResult").at(-1);
			expect(result).toMatchObject({ role: "toolResult", toolName: "first", isError: true });

			const current = getCurrentSystemMessage(harness.session.messages);
			expect(current?.toolsAdded?.map((value) => value.name)).toEqual(["second"]);
			expect(getSystemMessageText(current!)).toBe(harness.session.systemPrompt);
		} finally {
			harness.cleanup();
		}
	});

	test("setActiveTools in before_agent_start controls the same request", async () => {
		const extension: ExtensionFactory = (pi) => {
			for (const name of ["first", "second"]) {
				pi.registerTool({
					name,
					label: name,
					description: `${name} description`,
					parameters: Type.Object({}),
					execute: async () => ({ content: [{ type: "text", text: name }], details: {} }),
				});
			}
			let turn = 0;
			pi.on("before_agent_start", () => {
				if (turn++ === 1) pi.setActiveTools(["second"]);
			});
		};
		const harness = await createHarness({ extensionFactories: [extension], initialActiveToolNames: ["first"] });
		try {
			const requests: TranscriptContext[] = [];
			harness.setResponses([
				(providerContext) => {
					requests.push(providerContext);
					return fauxAssistantMessage("first");
				},
				(providerContext) => {
					requests.push(providerContext);
					return fauxAssistantMessage("second");
				},
			]);
			await harness.session.prompt("first");
			await harness.session.prompt("second");
			expect(requests).toHaveLength(2);
			const update = requests[1]?.messages.filter((message) => message.role === "system").at(-1);
			expect(update?.toolsRemoved).toEqual([{ name: "first" }]);
			expect(update?.toolsAdded).toBeUndefined();
			expect(harness.session.getActiveToolNames()).toEqual(["second"]);
		} finally {
			harness.cleanup();
		}
	});

	test("keeps tool declarations stable across a session JSON round-trip", async () => {
		const executableTool: AgentTool = {
			name: "plain",
			label: "Plain",
			description: "Plain tool",
			parameters: Type.Object({}),
			execute: async () => ({ content: [], details: {} }),
		};
		const harness = await createHarness({ tools: [executableTool], initialActiveToolNames: ["plain"] });
		try {
			harness.setResponses([fauxAssistantMessage("first"), fauxAssistantMessage("second")]);
			await harness.session.prompt("one");
			const head = harness.session.messages[0];
			if (head?.role !== "system") throw new Error("expected system message");
			const declaration = head.toolsAdded?.[0];
			if (!declaration) throw new Error("expected tool declaration");
			expect(Object.hasOwn(declaration, "constrainedSampling")).toBe(false);
			expect(Object.hasOwn(declaration, "execute")).toBe(false);

			// Simulate a resume: the persisted JSON must replay to the same declarations.
			harness.session.agent.state.messages = JSON.parse(JSON.stringify(harness.session.messages));
			await harness.session.prompt("two");
			expect(harness.session.messages.filter((message) => message.role === "system")).toHaveLength(1);
		} finally {
			harness.cleanup();
		}
	});
});
