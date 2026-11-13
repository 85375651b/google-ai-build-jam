import React, { useState, useEffect, useRef } from "react";
import { GenerationClient, isBusyStage, isPostprocessingStage } from "./generationClient.js";

const PERMISSION_LABELS = {
  FACTUAL_ONLY: "Factual Explanation Only (No Fictional Invention)",
  ANALYTICAL_ADVISORY: "Analytical & Consequence Review",
  IMPLEMENTATION_SCOPED: "Implementation Assistance (Author Premise)",
  BOUNDED_BRAINSTORMING: "Opt-In Scoped Brainstorming",
  EXPLICIT_DRAFTING: "Author-Authorized World Bible Drafting",
};

function renderInlineMarkdown(text) {
  if (!text) return null;
  const tokenRegex =
    /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`\n]+`|\[(?:Book-[^\]]+|Web-\d+)\])/g;
  const parts = String(text).split(tokenRegex);
  return parts.map((part, idx) => {
    if (!part) return null;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={idx}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={idx}>{part.slice(1, -1)}</code>;
    }
    if (/^\[(?:Book-[^\]]+|Web-\d+)\]$/.test(part)) {
      return (
        <span key={idx} className="md-inline-citation">
          {part}
        </span>
      );
    }
    return part;
  });
}

function FormattedMarkdown({ text }) {
  if (!text || !String(text).trim()) {
    return (
      <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
        No content written for this section yet.
      </span>
    );
  }
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    // Fenced code block
    if (trimmed.startsWith("```")) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      i += 1;
      elements.push(
        <pre key={`code-${i}`}>
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      elements.push(<hr key={`hr-${i}`} />);
      i += 1;
      continue;
    }

    // Headings
    const hMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (hMatch) {
      const level = hMatch[1].length;
      const content = renderInlineMarkdown(hMatch[2]);
      if (level === 1) elements.push(<h1 key={`h1-${i}`}>{content}</h1>);
      else if (level === 2) elements.push(<h2 key={`h2-${i}`}>{content}</h2>);
      else if (level === 3) elements.push(<h3 key={`h3-${i}`}>{content}</h3>);
      else elements.push(<h4 key={`h4-${i}`}>{content}</h4>);
      i += 1;
      continue;
    }

    // Blockquote
    if (trimmed.startsWith("> ")) {
      const quoteLines = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      elements.push(
        <blockquote key={`bq-${i}`}>
          {renderInlineMarkdown(quoteLines.join(" "))}
        </blockquote>
      );
      continue;
    }

    // Unordered list
    if (/^[-*•]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*•]\s+/, ""));
        i += 1;
      }
      elements.push(
        <ul key={`ul-${i}`}>
          {items.map((it, idx) => (
            <li key={idx}>{renderInlineMarkdown(it)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      elements.push(
        <ol key={`ol-${i}`}>
          {items.map((it, idx) => (
            <li key={idx}>{renderInlineMarkdown(it)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // Standard paragraph (collect consecutive non-special lines)
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith("> ") &&
      !/^[-*•]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !/^(-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i += 1;
    }
    elements.push(
      <p key={`p-${i}`}>{renderInlineMarkdown(paraLines.join("\n"))}</p>
    );
  }

  return <div className="md-rendered">{elements}</div>;
}

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [worlds, setWorlds] = useState([]);
  const [activeWorldId, setActiveWorldId] = useState("");
  const [activeWorldDetails, setActiveWorldDetails] = useState(null);
  const [activeTab, setActiveTab] = useState("chat"); // chat | bible | sources | settings

  // Chat state
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [requestStage, setRequestStage] = useState("idle");
  const [requestNotice, setRequestNotice] = useState("");
  const isStreaming = isBusyStage(requestStage);
  const [streamingText, setStreamingText] = useState("");
  const [streamMeta, setStreamMeta] = useState(null);
  const [chatError, setChatError] = useState("");
  const [expandedCitation, setExpandedCitation] = useState(null);
  const generationClientRef = useRef(null);
  const generationHandlers = useRef({});
  const scopeRef = useRef({});
  const displayedScopeRef = useRef({});
  const workspaceRevision = useRef(0);
  scopeRef.current = { worldId: activeWorldId, conversationId: activeConvId };
  const messagesEndRef = useRef(null);

  // World Bible & Proposals state
  const [sections, setSections] = useState([]);
  const [activeSectionId, setActiveSectionId] = useState("");
  const [editorTitle, setEditorTitle] = useState("");
  const [editorMarkdown, setEditorMarkdown] = useState("");
  const [sectionSearch, setSectionSearch] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [revisions, setRevisions] = useState([]);
  const [showRevisions, setShowRevisions] = useState(false);
  const [orphanedFactsPrompt, setOrphanedFactsPrompt] = useState([]);
  const [proposals, setProposals] = useState([]);
  const [editingProposalId, setEditingProposalId] = useState(null);
  const [editedProposalText, setEditedProposalText] = useState("");
  const [facts, setFacts] = useState([]);
  const [consistencyReport, setConsistencyReport] = useState(null);

  // Modals / Forms
  const [showNewWorldModal, setShowNewWorldModal] = useState(false);
  const [newWorldName, setNewWorldName] = useState("");
  const [newWorldDesc, setNewWorldDesc] = useState("");
  const [newWorldGenre, setNewWorldGenre] = useState("");
  const [newSectionTitle, setNewSectionTitle] = useState("");
  const [showOnboarding, setShowOnboarding] = useState(
    () => localStorage.getItem("wb_onboarding_dismissed") !== "true"
  );
  const [uploadingBook, setUploadingBook] = useState(false);
  const [modelInput, setModelInput] = useState("gpt-oss:20b");
  const [isRenamingWorld, setIsRenamingWorld] = useState(false);
  const [editWorldName, setEditWorldName] = useState("");
  const [editWorldDesc, setEditWorldDesc] = useState("");
  const [editWorldGenre, setEditWorldGenre] = useState("");
  const [worldSaveMsg, setWorldSaveMsg] = useState("");
  const [isRenamingThread, setIsRenamingThread] = useState(false);
  const [editThreadTitle, setEditThreadTitle] = useState("");
  const [showRollingSummary, setShowRollingSummary] = useState(false);
  const [lastFailedPrompt, setLastFailedPrompt] = useState("");
  const [modelSaveMsg, setModelSaveMsg] = useState("");
  const [bibleViewMode, setBibleViewMode] = useState("edit"); // edit | split | preview
  const [importMsg, setImportMsg] = useState("");
  const [darkMode, setDarkMode] = useState(
    () => localStorage.getItem("wb_theme") !== "light"
  );

  const matchesScope = (owner) => scopeRef.current.worldId === owner.worldId && scopeRef.current.conversationId === owner.conversationId;
  const showSavedAnswer = (payload, owner) => {
    if (!payload.answer_saved || !payload.assistant_message) return;
    owner.answerSaved = true;
    setMessages((previous) => [
      ...previous.filter((message) => message.id !== payload.assistant_message.id).map((message) =>
        message.id === owner.tempId ? { ...message, id: payload.user_message_id } : message),
      payload.assistant_message,
    ]);
    setStreamingText("");
  };
  generationHandlers.current = {
    onState(stage, message, owner) {
      if (!matchesScope(owner)) return;
      setRequestStage(isBusyStage(stage) ? stage : "idle");
      setRequestNotice(stage === "complete" ? "Finished." : message);
    },
    onEvent(payload, owner) {
      if (!matchesScope(owner)) return;
      if (payload.type === "turn_meta") setStreamMeta(payload);
      else if (payload.type === "delta") setStreamingText((previous) => previous + payload.text);
      else if (payload.type === "answer_completed") setStreamingText(payload.text);
      else if (payload.type === "response_saved") showSavedAnswer(payload, owner);
      else if (["complete", "cancelled", "error"].includes(payload.type)) {
        showSavedAnswer(payload, owner);
        if (!payload.answer_saved) setMessages((previous) => previous.filter((message) => message.id !== owner.tempId));
        setStreamingText("");
        setStreamMeta(null);
        if (payload.type === "error") {
          setChatError(payload.message || "The request failed.");
          setLastFailedPrompt(owner.content);
        }
        if (payload.type === "complete") {
          loadWorldWorkspace(owner.worldId);
          fetchMessages(owner.conversationId);
          fetchSystemStatus();
        }
      }
    },
    onFailure(error, owner) {
      if (!matchesScope(owner)) return;
      setMessages((previous) => previous.filter((message) => message.id !== owner.tempId));
      setStreamingText("");
      setStreamMeta(null);
      setChatError(error.message || "The request was interrupted.");
      setLastFailedPrompt(owner.content);
      fetchMessages(owner.conversationId);
    },
  };
  if (!generationClientRef.current) {
    generationClientRef.current = new GenerationClient({
      onState: (...args) => generationHandlers.current.onState(...args),
      onEvent: (...args) => generationHandlers.current.onEvent(...args),
      onFailure: (...args) => generationHandlers.current.onFailure(...args),
    });
  }

  useEffect(() => {
    const client = generationClientRef.current;
    const owner = client.active;
    const previous = displayedScopeRef.current;
    const changed = previous.worldId !== activeWorldId || previous.conversationId !== activeConvId;
    displayedScopeRef.current = { worldId: activeWorldId, conversationId: activeConvId };
    if (owner && !matchesScope(owner)) client.detach();
    if (changed && (!owner || !matchesScope(owner))) {
      workspaceRevision.current += 1;
      setRequestStage("idle");
      setStreamingText("");
      setStreamMeta(null);
      setRequestNotice("");
      setChatError("");
      setMessages([]);
    }
  }, [activeWorldId, activeConvId]);

  useEffect(() => {
    const detach = () => generationClientRef.current.detach();
    window.addEventListener("pagehide", detach);
    return () => {
      window.removeEventListener("pagehide", detach);
      detach();
    };
  }, []);

  useEffect(() => {
    const theme = darkMode ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("wb_theme", theme);
  }, [darkMode]);

  useEffect(() => {
    fetchSystemStatus();
    fetchWorlds();
  }, []);

  useEffect(() => {
    if (activeWorldId) {
      loadWorldWorkspace(activeWorldId);
    }
  }, [activeWorldId]);

  useEffect(() => {
    if (activeConvId) {
      fetchMessages(activeConvId);
    }
  }, [activeConvId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/system/status");
      const data = await res.json();
      setSystemStatus(data);
      if (data.local_model) {
        setModelInput(data.local_model);
      }
    } catch (err) {
      console.error("Failed loading system status:", err);
    }
  }

  async function fetchWorlds() {
    try {
      const res = await fetch("/api/worlds");
      const data = await res.json();
      setWorlds(data);
      if (data.length > 0 && !activeWorldId) {
        setActiveWorldId(data[0].id);
      } else if (data.length === 0) {
        setShowNewWorldModal(true);
      }
    } catch (err) {
      console.error("Failed loading worlds:", err);
    }
  }

  async function loadWorldWorkspace(worldId) {
    const revision = workspaceRevision.current;
    try {
      const [detailsRes, secsRes, convsRes, propsRes, factsRes, consRes] =
        await Promise.all([
          fetch(`/api/worlds/${worldId}`),
          fetch(`/api/worlds/${worldId}/bible/sections`),
          fetch(`/api/worlds/${worldId}/conversations`),
          fetch(`/api/worlds/${worldId}/proposals?status=PENDING`),
          fetch(`/api/worlds/${worldId}/facts`),
          fetch(`/api/worlds/${worldId}/consistency`),
        ]);

      const details = await detailsRes.json();
      const secs = await secsRes.json();
      const convs = await convsRes.json();
      const props = await propsRes.json();
      const factsData = await factsRes.json();
      const consData = await consRes.json();
      if (scopeRef.current.worldId !== worldId || workspaceRevision.current !== revision) return;

      setActiveWorldDetails(details);
      setEditWorldName(details.name || "");
      setEditWorldDesc(details.description || "");
      setEditWorldGenre(details.genre_notes || "");
      setIsRenamingWorld(false);
      setSections(secs);
      setProposals(props);
      setFacts(factsData);
      setConsistencyReport(consData);

      if (secs.length > 0) {
        const keepSec =
          secs.find((s) => s.id === activeSectionId) || secs[0];
        setActiveSectionId(keepSec.id);
        setEditorTitle(keepSec.title);
        setEditorMarkdown(keepSec.markdown_content || "");
      } else {
        setActiveSectionId("");
        setEditorTitle("");
        setEditorMarkdown("");
      }

      setConversations(convs);
      if (convs.length > 0) {
        setActiveConvId((prevConvId) => {
          if (prevConvId && convs.some((c) => c.id === prevConvId)) {
            return prevConvId;
          }
          return convs[0].id;
        });
      } else {
        setActiveConvId("");
        setMessages([]);
      }
    } catch (err) {
      console.error("Error loading world workspace:", err);
    }
  }

  async function fetchMessages(convId) {
    const revision = workspaceRevision.current;
    try {
      const res = await fetch(`/api/conversations/${convId}/messages`);
      const data = await res.json();
      if (scopeRef.current.conversationId !== convId || workspaceRevision.current !== revision) return;
      setMessages(data);
    } catch (err) {
      console.error("Failed loading messages:", err);
    }
  }

  function selectBibleSection(sec) {
    setActiveSectionId(sec.id);
    setEditorTitle(sec.title);
    setEditorMarkdown(sec.markdown_content || "");
    setShowRevisions(false);
    setSaveMessage("");
  }

  async function handleCreateWorld(e) {
    e.preventDefault();
    if (!newWorldName.trim()) return;
    const res = await fetch("/api/worlds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: newWorldName.trim(),
        description: newWorldDesc.trim(),
        genre_notes: newWorldGenre.trim(),
        seed_default_subjects: true,
      }),
    });
    if (res.ok) {
      const created = await res.json();
      setNewWorldName("");
      setNewWorldDesc("");
      setNewWorldGenre("");
      setShowNewWorldModal(false);
      await fetchWorlds();
      setActiveWorldId(created.id);
    }
  }

  async function handleUpdateWorldDetails(e) {
    if (e) e.preventDefault();
    if (!activeWorldId || !editWorldName.trim()) return;
    const res = await fetch(`/api/worlds/${activeWorldId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: editWorldName.trim(),
        description: editWorldDesc.trim(),
        genre_notes: editWorldGenre.trim(),
      }),
    });
    if (res.ok) {
      setIsRenamingWorld(false);
      setWorldSaveMsg("Saved!");
      setTimeout(() => setWorldSaveMsg(""), 2500);
      await fetchWorlds();
      await loadWorldWorkspace(activeWorldId);
    }
  }

  async function handleDeleteWorld() {
    if (!activeWorldId) return;
    const confirmed = window.confirm(
      `Delete world "${activeWorldDetails?.name}" and its entire World Bible and conversation history? This cannot be undone.`
    );
    if (!confirmed) return;
    await fetch(`/api/worlds/${activeWorldId}`, { method: "DELETE" });
    setActiveWorldId("");
    setActiveWorldDetails(null);
    await fetchWorlds();
  }

  async function handleCreateConversation() {
    if (!activeWorldId) return;
    const defaultTitle = `Consultation #${conversations.length + 1}`;
    const res = await fetch(`/api/worlds/${activeWorldId}/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: defaultTitle }),
    });
    if (res.ok) {
      const created = await res.json();
      const convsRes = await fetch(`/api/worlds/${activeWorldId}/conversations`);
      const convs = await convsRes.json();
      setConversations(convs);
      setActiveConvId(created.id);
      setIsRenamingThread(false);
      setShowRollingSummary(false);
    }
  }

  async function handleRenameConversation(e) {
    if (e) e.preventDefault();
    if (!activeConvId || !editThreadTitle.trim()) return;
    const res = await fetch(`/api/conversations/${activeConvId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: editThreadTitle.trim() }),
    });
    if (res.ok) {
      setIsRenamingThread(false);
      const convsRes = await fetch(`/api/worlds/${activeWorldId}/conversations`);
      setConversations(await convsRes.json());
    }
  }

  async function handleArchiveConversation() {
    if (!activeConvId || !activeWorldId) return;
    const currentConv = conversations.find((c) => c.id === activeConvId);
    const confirmed = window.confirm(
      `Archive conversation thread "${currentConv?.title || "Current Thread"}"?`
    );
    if (!confirmed) return;
    const res = await fetch(`/api/conversations/${activeConvId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ is_archived: true }),
    });
    if (res.ok) {
      const data = await res.json();
      const convsRes = await fetch(`/api/worlds/${activeWorldId}/conversations`);
      const convs = await convsRes.json();
      setConversations(convs);
      const nextId =
        data.replacement_conversation_id || (convs.length > 0 ? convs[0].id : "");
      setActiveConvId(nextId);
      setIsRenamingThread(false);
      setShowRollingSummary(false);
    }
  }

  async function handleSendMessage(e, retryOverride = "") {
    if (e) e.preventDefault();
    const promptText = (retryOverride || chatInput).trim();
    const client = generationClientRef.current;
    if (!promptText || !activeConvId || client.active) return;
    workspaceRevision.current += 1;
    if (!retryOverride) setChatInput("");
    setChatError("");
    setLastFailedPrompt("");
    setStreamingText("");
    setStreamMeta(null);
    const tempId = "temp-user-" + crypto.randomUUID();
    setMessages((previous) => [...previous, {
      id: tempId, role: "USER", content: promptText, created_at: new Date().toISOString(),
    }]);
    await client.start({ content: promptText, conversationId: activeConvId, worldId: activeWorldId, tempId });
  }

  function handleStopGeneration() {
    void generationClientRef.current.cancel();
  }

  async function handleProposalDecision(proposalId, action) {
    const body = { action };
    if (action === "EDIT_AND_APPROVE") {
      body.edited_content = editedProposalText;
    }
    const res = await fetch(
      `/api/worlds/${activeWorldId}/proposals/${proposalId}/resolve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );
    if (res.ok) {
      setEditingProposalId(null);
      await loadWorldWorkspace(activeWorldId);
    }
  }

  async function handleSaveBibleSection() {
    if (!activeWorldId || !activeSectionId) return;
    setSaveMessage("Saving...");
    const currentSec = sections.find((s) => s.id === activeSectionId);
    const res = await fetch(
      `/api/worlds/${activeWorldId}/bible/sections/${activeSectionId}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: editorTitle,
          markdown_content: editorMarkdown,
          expected_version_number: currentSec?.version_number,
          author_note: "Direct edit in World Bible workspace",
        }),
      }
    );
    if (res.ok) {
      const data = await res.json();
      setSaveMessage(`Saved v${data.section.version_number} (Author Authority)`);
      if (
        data.orphaned_facts_requiring_confirmation &&
        data.orphaned_facts_requiring_confirmation.length > 0
      ) {
        setOrphanedFactsPrompt(data.orphaned_facts_requiring_confirmation);
      }
      const secsRes = await fetch(`/api/worlds/${activeWorldId}/bible/sections`);
      setSections(await secsRes.json());
      const factsRes = await fetch(`/api/worlds/${activeWorldId}/facts`);
      setFacts(await factsRes.json());
    } else {
      const err = await res.json().catch(() => ({}));
      setSaveMessage(`Error: ${err.detail || "Save failed"}`);
    }
  }

  async function handleResolveOrphanedFacts(action) {
    const factIds = orphanedFactsPrompt.map((f) => f.id);
    await fetch(`/api/worlds/${activeWorldId}/bible/orphaned-facts/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fact_ids: factIds, action }),
    });
    setOrphanedFactsPrompt([]);
    await loadWorldWorkspace(activeWorldId);
  }

  async function handleCreateSection(e) {
    e.preventDefault();
    if (!newSectionTitle.trim() || !activeWorldId) return;
    const res = await fetch(`/api/worlds/${activeWorldId}/bible/sections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: newSectionTitle.trim(), markdown_content: "" }),
    });
    if (res.ok) {
      const created = await res.json();
      setNewSectionTitle("");
      const secsRes = await fetch(`/api/worlds/${activeWorldId}/bible/sections`);
      const updatedSecs = await secsRes.json();
      setSections(updatedSecs);
      selectBibleSection(created);
    }
  }

  async function handleDeleteSection(secId) {
    if (!window.confirm("Delete this World Bible section?")) return;
    await fetch(`/api/worlds/${activeWorldId}/bible/sections/${secId}`, {
      method: "DELETE",
    });
    await loadWorldWorkspace(activeWorldId);
  }

  async function handleLoadRevisions() {
    if (!activeSectionId) return;
    const res = await fetch(
      `/api/worlds/${activeWorldId}/bible/sections/${activeSectionId}/revisions`
    );
    if (res.ok) {
      setRevisions(await res.json());
      setShowRevisions(true);
    }
  }

  async function handleRevertRevision(versionNumber) {
    const res = await fetch(
      `/api/worlds/${activeWorldId}/bible/sections/${activeSectionId}/revert/${versionNumber}`,
      { method: "POST" }
    );
    if (res.ok) {
      const reverted = await res.json();
      setEditorMarkdown(reverted.markdown_content);
      setShowRevisions(false);
      await loadWorldWorkspace(activeWorldId);
    }
  }

  async function handleBookUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadingBook(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      await fetch(`/api/sources/upload?title=${encodeURIComponent(file.name)}`, {
        method: "POST",
        body: formData,
      });
      await fetchSystemStatus();
    } finally {
      setUploadingBook(false);
    }
  }

  async function handleExpertiseOverride(domainKey, level) {
    await fetch(`/api/worlds/${activeWorldId}/expertise`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain_key: domainKey, expertise_level: level }),
    });
    await loadWorldWorkspace(activeWorldId);
  }

  async function handleResolveConsistencyFlag(factId) {
    if (!activeWorldId || !factId) return;
    const res = await fetch(
      `/api/worlds/${activeWorldId}/consistency/facts/${factId}/resolve`,
      { method: "POST" }
    );
    if (res.ok) {
      await loadWorldWorkspace(activeWorldId);
    }
  }

  async function handleUpdateModel(e) {
    e.preventDefault();
    const res = await fetch("/api/system/model", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ local_model: modelInput }),
    });
    if (res.ok) {
      setModelSaveMsg("Model updated!");
      setTimeout(() => setModelSaveMsg(""), 2500);
      await fetchSystemStatus();
    } else {
      setModelSaveMsg("Unable to update model. Enter an installed Ollama model name.");
    }
  }

  async function handleImportProjectBundle(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImportMsg("Importing project archive...");
    try {
      const rawText = await file.text();
      const bundle = JSON.parse(rawText);
      const res = await fetch("/api/worlds/import-project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bundle),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Project import failed");
      }
      const restored = await res.json();
      await fetchWorlds();
      setActiveWorldId(restored.id);
      setImportMsg(`Restored "${restored.name}"!`);
      setTimeout(() => setImportMsg(""), 3500);
    } catch (err) {
      setImportMsg(`Error: ${err.message || "Invalid project JSON"}`);
    } finally {
      e.target.value = "";
    }
  }

  const filteredSections = sections.filter(
    (s) =>
      s.title.toLowerCase().includes(sectionSearch.toLowerCase()) ||
      (s.markdown_content || "").toLowerCase().includes(sectionSearch.toLowerCase())
  );

  return (
    <div className="app-shell" data-theme={darkMode ? "dark" : "light"}>
      {/* Persistent Navigation & Project Sidebar */}
      <aside className="sidebar" aria-label="Project and Workspace Navigation">
        <div className="brand-header">
          <div className="brand-title">Worldbuilding Studio</div>
          <div className="brand-subtitle">Author-First Research & Canon Consultant</div>
        </div>

        <div className="sidebar-section-label">Active Fictional World</div>
        <select
          aria-label="Select Active World Project"
          className="world-selector"
          value={activeWorldId}
          onChange={(e) => setActiveWorldId(e.target.value)}
        >
          {worlds.map((w) => (
            <option key={w.id} value={w.id}>
              {w.name}
            </option>
          ))}
        </select>
        <button
          className="btn btn-secondary"
          style={{ width: "100%", marginBottom: "6px", fontSize: "0.8rem" }}
          onClick={() => setShowNewWorldModal(true)}
        >
          + New Fictional World
        </button>
        <label
          className="btn btn-secondary"
          style={{
            width: "100%",
            marginBottom: "10px",
            fontSize: "0.78rem",
            textAlign: "center",
            display: "block",
          }}
          title="Restore a full world project from a .JSON backup archive"
        >
          Import Project (.JSON)
          <input
            type="file"
            accept=".json,application/json"
            style={{ display: "none" }}
            onChange={handleImportProjectBundle}
          />
        </label>
        {importMsg && (
          <div style={{ fontSize: "0.75rem", color: "#a7f3d0", marginBottom: "8px" }}>
            {importMsg}
          </div>
        )}

        <div className="sidebar-section-label">Workspace Areas</div>
        <button
          className={`nav-btn ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          <span>1. Consultant Chat</span>
          {proposals.length > 0 && (
            <span className="badge-pill badge-amber" title="Pending World Bible updates">
              {proposals.length}
            </span>
          )}
        </button>
        <button
          className={`nav-btn ${activeTab === "bible" ? "active" : ""}`}
          onClick={() => setActiveTab("bible")}
        >
          <span>2. World Bible</span>
          <span style={{ fontSize: "0.75rem", opacity: 0.75 }}>{sections.length}</span>
        </button>
        <button
          className={`nav-btn ${activeTab === "sources" ? "active" : ""}`}
          onClick={() => setActiveTab("sources")}
        >
          <span>3. Sources & Research</span>
        </button>
        <button
          className={`nav-btn ${activeTab === "settings" ? "active" : ""}`}
          onClick={() => setActiveTab("settings")}
        >
          <span>4. Project Settings</span>
        </button>

        <div style={{ marginTop: "auto", paddingTop: "14px", fontSize: "0.76rem", color: "#9da7a2" }}>
          <div>
            <span
              className={`status-dot ${
                systemStatus?.local_model_status?.ready ? "status-ok" : "status-warn"
              }`}
            />
            Local model:{" "}
            {systemStatus?.local_model_status?.ready
              ? `Ready (${systemStatus.local_model || "gpt-oss:20b"})`
              : (systemStatus?.local_model_status?.message || "Checking Ollama…")}
          </div>
          <div style={{ marginTop: "4px" }}>
            <span
              className={`status-dot ${
                systemStatus?.reference_documents?.length > 0 ? "status-ok" : "status-warn"
              }`}
            />
            Book RAG:{" "}
            {systemStatus?.reference_documents?.[0]
              ? `${systemStatus.reference_documents[0].total_chunks} passages`
              : "Indexing..."}
          </div>
          <div style={{ marginTop: "4px" }}>
            <span
              className={`status-dot ${
                systemStatus?.has_valid_tavily_key ? "status-ok" : "status-warn"
              }`}
            />
            Tavily Search:{" "}
            {systemStatus?.has_valid_tavily_key ? "Ready (.env)" : "Optional (.env)"}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-area">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-world-info">
            {isRenamingWorld ? (
              <form
                onSubmit={handleUpdateWorldDetails}
                style={{ display: "flex", gap: "8px", alignItems: "center" }}
              >
                <input
                  className="input-field"
                  style={{ width: "220px", fontWeight: 700 }}
                  value={editWorldName}
                  onChange={(e) => setEditWorldName(e.target.value)}
                  placeholder="World name..."
                  autoFocus
                />
                <input
                  className="input-field"
                  style={{ width: "260px" }}
                  value={editWorldDesc}
                  onChange={(e) => setEditWorldDesc(e.target.value)}
                  placeholder="Brief description (optional)..."
                />
                <button type="submit" className="btn btn-primary" style={{ padding: "6px 12px" }}>
                  Save
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: "6px 10px" }}
                  onClick={() => {
                    setEditWorldName(activeWorldDetails?.name || "");
                    setEditWorldDesc(activeWorldDetails?.description || "");
                    setIsRenamingWorld(false);
                  }}
                >
                  Cancel
                </button>
              </form>
            ) : (
              <>
                <span className="topbar-world-name">
                  {activeWorldDetails?.name || "No World Selected"}
                </span>
                {activeWorldDetails?.description && (
                  <span style={{ fontSize: "0.84rem", color: "var(--text-secondary)" }}>
                    — {activeWorldDetails.description}
                  </span>
                )}
                {activeWorldId && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ padding: "3px 9px", fontSize: "0.75rem" }}
                    onClick={() => setIsRenamingWorld(true)}
                  >
                    Rename
                  </button>
                )}
              </>
            )}
          </div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setDarkMode((prev) => !prev)}
              title="Toggle Dark / Light Mode"
            >
              {darkMode ? "☀ Light Mode" : "☾ Dark Mode"}
            </button>
            {activeWorldId && (
              <>
                <a
                  className="btn btn-secondary"
                  href={`/api/worlds/${activeWorldId}/bible/export?fmt=md`}
                  download
                >
                  Export .MD
                </a>
                <a
                  className="btn btn-secondary"
                  href={`/api/worlds/${activeWorldId}/bible/export?fmt=pdf`}
                  download
                >
                  Export .PDF
                </a>
                <a
                  className="btn btn-secondary"
                  href={`/api/worlds/${activeWorldId}/bible/export?fmt=docx`}
                  download
                >
                  Export .DOCX
                </a>
                <a
                  className="btn btn-secondary"
                  href={`/api/worlds/${activeWorldId}/export-project`}
                  download
                  title="Download complete portable project backup (.JSON) including World Bible, revisions, facts, and conversations"
                >
                  Backup .JSON
                </a>
              </>
            )}
          </div>
        </header>

        {/* Skippable First-Run Onboarding Banner */}
        {showOnboarding && (
          <div
            style={{
              background: "var(--accent-blue-bg)",
              borderBottom: "1px solid var(--accent-blue-border)",
              color: "var(--text-primary)",
              padding: "10px 24px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "0.84rem",
            }}
          >
            <div>
              <strong>Welcome to the Worldbuilding Consultant:</strong> (1) Create or select a
              world project, (2) Chat naturally—the agent explains real-world systems and helps
              implement your ideas without taking creative control, (3) Review proposed canon
              updates or edit your <strong>World Bible</strong> directly at any time.
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: "4px 10px", fontSize: "0.78rem" }}
              onClick={() => {
                localStorage.setItem("wb_onboarding_dismissed", "true");
                setShowOnboarding(false);
              }}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Staged World Bible Proposals Require Explicit Author Approval */}
        {proposals.length > 0 && (
          <div className="proposal-banner" role="region" aria-label="Pending World Bible Proposals">
            <div style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--accent-amber)" }}>
              PROPOSED WORLD BIBLE UPDATES ({proposals.length}) — Nothing becomes canon without your
              explicit approval:
            </div>
            {proposals.map((p) => (
              <div key={p.id} className="proposal-card">
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                    <strong>{p.proposed_title}</strong>
                    <span
                      className={`provenance-badge ${
                        p.proposed_provenance === "AUTHOR_CREATED" ? "prov-author" : "prov-ai"
                      }`}
                    >
                      {p.proposed_provenance === "AUTHOR_CREATED"
                        ? "Author Established"
                        : "AI Suggested (Requires Approval)"}
                    </span>
                  </div>
                  {editingProposalId === p.id ? (
                    <textarea
                      className="input-field"
                      style={{ marginTop: "6px", minHeight: "65px" }}
                      value={editedProposalText}
                      onChange={(e) => setEditedProposalText(e.target.value)}
                    />
                  ) : (
                    <div style={{ fontSize: "0.88rem", marginTop: "4px" }}>
                      {p.proposed_content}
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", gap: "6px" }}>
                  {editingProposalId === p.id ? (
                    <button
                      className="btn btn-primary"
                      onClick={() => handleProposalDecision(p.id, "EDIT_AND_APPROVE")}
                    >
                      Save & Approve
                    </button>
                  ) : (
                    <>
                      <button
                        className="btn btn-primary"
                        onClick={() => handleProposalDecision(p.id, "APPROVE")}
                      >
                        Approve
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => {
                          setEditingProposalId(p.id);
                          setEditedProposalText(p.proposed_content);
                        }}
                      >
                        Edit
                      </button>
                      <button
                        className="btn btn-danger"
                        onClick={() => handleProposalDecision(p.id, "REJECT")}
                      >
                        Reject
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Decision 3(b) Orphaned Fact Confirmation Banner */}
        {orphanedFactsPrompt.length > 0 && (
          <div className="proposal-banner">
            <div style={{ fontSize: "0.85rem", fontWeight: 600 }}>
              You deleted prose in the World Bible linked to {orphanedFactsPrompt.length} established
              fact(s). Should the agent also retire these facts from active canon?
            </div>
            <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
              <button
                className="btn btn-danger"
                onClick={() => handleResolveOrphanedFacts("RETIRE")}
              >
                Retire Removed Facts (Supersede)
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => handleResolveOrphanedFacts("KEEP_ACTIVE")}
              >
                Keep as Background Canon
              </button>
            </div>
          </div>
        )}

        {/* Workspace Tab Views */}
        <div className="workspace-body">
          {/* TAB 1: CHAT INTERFACE */}
          {activeTab === "chat" && (
            <div className="chat-column">
              {/* Multi-Thread Conversation Bar */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: "10px",
                  padding: "10px 24px",
                  borderBottom: "1px solid var(--border-subtle)",
                  background: "var(--bg-surface)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                  <span
                    style={{
                      fontSize: "0.76rem",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.04em",
                      color: "var(--text-muted)",
                    }}
                  >
                    Thread:
                  </span>
                  {isRenamingThread ? (
                    <form
                      onSubmit={handleRenameConversation}
                      style={{ display: "flex", gap: "6px", alignItems: "center" }}
                    >
                      <input
                        className="input-field"
                        style={{ width: "220px", padding: "4px 8px", fontSize: "0.84rem" }}
                        value={editThreadTitle}
                        onChange={(e) => setEditThreadTitle(e.target.value)}
                        placeholder="Thread title..."
                        autoFocus
                      />
                      <button
                        type="submit"
                        className="btn btn-primary"
                        style={{ padding: "4px 10px", fontSize: "0.76rem" }}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: "4px 8px", fontSize: "0.76rem" }}
                        onClick={() => setIsRenamingThread(false)}
                      >
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <>
                      <select
                        aria-label="Select Conversation Thread"
                        className="input-field"
                        style={{
                          width: "auto",
                          minWidth: "220px",
                          padding: "5px 10px",
                          fontSize: "0.84rem",
                          fontWeight: 600,
                        }}
                        value={activeConvId}
                        onChange={(e) => {
                          setActiveConvId(e.target.value);
                          setIsRenamingThread(false);
                          setShowRollingSummary(false);
                        }}
                      >
                        {conversations.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.title}
                          </option>
                        ))}
                      </select>
                      {activeConvId && (
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{ padding: "4px 9px", fontSize: "0.75rem" }}
                          onClick={() => {
                            const curr = conversations.find((c) => c.id === activeConvId);
                            setEditThreadTitle(curr?.title || "");
                            setIsRenamingThread(true);
                          }}
                        >
                          Rename Thread
                        </button>
                      )}
                    </>
                  )}
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  {conversations.find((c) => c.id === activeConvId)?.rolling_summary && (
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{
                        padding: "4px 10px",
                        fontSize: "0.75rem",
                        background: "var(--success-bg)",
                        borderColor: "var(--success-border)",
                        color: "var(--success-text)",
                      }}
                      onClick={() => setShowRollingSummary((prev) => !prev)}
                      title="View condensed rolling summary of turns older than the 12-message window"
                    >
                      {showRollingSummary ? "Hide Rolling Summary" : "Rolling Memory Active"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ padding: "4px 10px", fontSize: "0.76rem" }}
                    onClick={handleCreateConversation}
                  >
                    + New Thread
                  </button>
                  {activeConvId && (
                    <button
                      type="button"
                      className="btn btn-danger"
                      style={{ padding: "4px 10px", fontSize: "0.76rem" }}
                      onClick={handleArchiveConversation}
                    >
                      Archive Thread
                    </button>
                  )}
                </div>
              </div>

              {showRollingSummary &&
                conversations.find((c) => c.id === activeConvId)?.rolling_summary && (
                  <div
                    style={{
                      padding: "10px 24px",
                      background: "var(--success-bg)",
                      borderBottom: "1px solid var(--success-border)",
                      fontSize: "0.82rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <strong style={{ color: "var(--success-text)" }}>
                      Rolling Conversation Summary (Turns Older Than 12 Messages):
                    </strong>
                    <div style={{ marginTop: "4px", whiteSpace: "pre-wrap" }}>
                      {conversations.find((c) => c.id === activeConvId)?.rolling_summary}
                    </div>
                  </div>
                )}

              <div className="messages-scroll">
                {messages.length === 0 && !isStreaming && (
                  <div className="card-panel" style={{ marginTop: "20px" }}>
                    <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                      Consult with your Worldbuilding Assistant
                    </h3>
                    <p style={{ color: "var(--text-secondary)", fontSize: "0.92rem" }}>
                      Describe what you are trying to accomplish in ordinary language—no technical
                      vocabulary or AI prompting skills required. You remain in complete creative
                      control:
                    </p>
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "8px",
                        marginTop: "12px",
                      }}
                    >
                      <button
                        type="button"
                        className="starter-chip"
                        onClick={() =>
                          setChatInput(
                            "How did medieval taxation and tribute collection work before standing bureaucracies?"
                          )
                        }
                      >
                        <strong>1. Ask how real-world systems work:</strong>{" "}
                        <em>
                          &ldquo;How did medieval taxation and tribute collection work before
                          standing bureaucracies?&rdquo;
                        </em>
                      </button>
                      <button
                        type="button"
                        className="starter-chip"
                        onClick={() =>
                          setChatInput(
                            "I want a government where citizens elect their leaders, but only landowners who completed naval service can vote. How would this function in practice?"
                          )
                        }
                      >
                        <strong>2. Implement your own creative concept:</strong>{" "}
                        <em>
                          &ldquo;I want a government where citizens elect their leaders, but only
                          landowners who completed naval service can vote.&rdquo;
                        </em>
                      </button>
                      <button
                        type="button"
                        className="starter-chip"
                        onClick={() =>
                          setChatInput(
                            "Give me three possible ways this council could handle succession without hereditary monarchy."
                          )
                        }
                      >
                        <strong>3. Request opt-in brainstorming when you want options:</strong>{" "}
                        <em>
                          &ldquo;Give me three possible ways this council could handle succession
                          without hereditary monarchy.&rdquo;
                        </em>
                      </button>
                    </div>
                  </div>
                )}

                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`msg-card ${m.role === "USER" ? "msg-user" : "msg-assistant"}`}
                  >
                    {m.role === "ASSISTANT" && m.metadata && (
                      <div className="msg-meta-bar">
                        <span
                          className={`meta-tag ${
                            m.permission_level === "BOUNDED_BRAINSTORMING"
                              ? "perm-brainstorm"
                              : "perm-strict"
                          }`}
                        >
                          Mode: {PERMISSION_LABELS[m.permission_level] || m.permission_level}
                        </span>
                        {m.metadata.naming_mode &&
                          m.metadata.naming_mode !== "NONE" && (
                            <span className="meta-tag perm-brainstorm">
                              Naming:{" "}
                              {m.metadata.naming_mode === "MINOR_QUICK"
                                ? "Quick / Minor Naming"
                                : "Context-Driven Major Naming"}
                            </span>
                          )}
                        {m.metadata.book_coverage && (
                          <span className="meta-tag">
                            Book Coverage: {m.metadata.book_coverage}
                          </span>
                        )}
                      </div>
                    )}

                    {m.role === "ASSISTANT" && m.metadata?.request_state && m.metadata.request_state !== "complete" && (
                      <div className="msg-meta-bar" role="status">
                        {m.metadata.request_state === "answer_saved" ? "Answer saved; remaining updates are pending." : `Request ${m.metadata.request_state}; completed answer retained.`}
                      </div>
                    )}
                    {m.role === "ASSISTANT" ? (
                      <FormattedMarkdown text={m.content} />
                    ) : (
                      <div>{m.content}</div>
                    )}

                    {m.role === "ASSISTANT" && m.metadata?.canon_result && (
                      <div className="citation-box" aria-label="World Bible update outcome">
                        <strong>World Bible: {({ committed: "Saved", proposed: "Awaiting approval", clarification_required: "Needs clarification", error: "Not saved", failed: "Not saved", cancelled: "Cancelled" })[m.metadata.canon_result.status] || "Review required"}</strong>
                        {!!m.metadata.canon_result.operations?.length && (
                          <ul>
                            {m.metadata.canon_result.operations.map((operation, index) => {
                              const change = operation.operation || operation;
                              return <li key={index}>
                                {change.entity_ref || "Proposed change"}
                                {change.property_key ? `: ${change.property_key.replaceAll("_", " ")}` : ""}
                                {change.value !== undefined ? ` → ${typeof change.value === "string" ? change.value : JSON.stringify(change.value)}` : ""}
                                {" — "}{operation.state === "committed" ? "Saved" : operation.state === "proposed" ? "Awaiting approval" : "Not applied"}
                              </li>;
                            })}
                          </ul>
                        )}
                      </div>
                    )}

                    {m.role === "ASSISTANT" &&
                      m.metadata?.updated_bible_sections &&
                      m.metadata.updated_bible_sections.length > 0 && (
                        <div
                          style={{
                            marginTop: "10px",
                            padding: "10px 12px",
                            background: "var(--success-bg)",
                            border: "1px solid var(--success-border)",
                            borderRadius: "6px",
                            fontSize: "0.82rem",
                          }}
                        >
                          <div style={{ fontWeight: 700, color: "var(--success-text)", marginBottom: "6px" }}>
                            ✓ World Bible Automatically Updated (
                            {m.metadata.updated_bible_sections.length} section
                            {m.metadata.updated_bible_sections.length > 1 ? "s" : ""}):
                          </div>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                            {m.metadata.updated_bible_sections.map((uSec, idx) => (
                              <button
                                key={idx}
                                type="button"
                                className="btn btn-secondary"
                                style={{ padding: "4px 10px", fontSize: "0.78rem" }}
                                onClick={() => {
                                  const match = sections.find(
                                    (s) => s.id === uSec.id || s.slug === uSec.slug
                                  );
                                  if (match) selectBibleSection(match);
                                  setActiveTab("bible");
                                }}
                              >
                                View {uSec.title} (v{uSec.version_number}) →
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                    {m.role === "ASSISTANT" &&
                      m.metadata?.citations &&
                      m.metadata.citations.length > 0 && (
                        <div className="citation-box">
                          <strong>Verified Sources Used: </strong>
                          {m.metadata.citations.map((cit, idx) => (
                            <button
                              key={idx}
                              type="button"
                              className="citation-chip"
                              onClick={() =>
                                setExpandedCitation(
                                  expandedCitation === `${m.id}-${idx}` ? null : `${m.id}-${idx}`
                                )
                              }
                            >
                              <span>
                                [{cit.citation_id}] {cit.label}
                              </span>
                            </button>
                          ))}
                          {m.metadata.citations.map(
                            (cit, idx) =>
                              expandedCitation === `${m.id}-${idx}` && (
                                <div
                                  key={`exp-${idx}`}
                                  style={{
                                    background: "var(--bg-subtle)",
                                    border: "1px solid var(--border-subtle)",
                                    borderRadius: "6px",
                                    padding: "8px 10px",
                                    marginTop: "6px",
                                    fontSize: "0.8rem",
                                  }}
                                >
                                  <div>
                                    <strong>
                                      {cit.source_type === "REFERENCE_BOOK"
                                        ? "Primary Reference Book"
                                        : "Supplementary Online Research"}
                                      :
                                    </strong>{" "}
                                    {cit.label}
                                  </div>
                                  {cit.url && (
                                    <div>
                                      <a href={cit.url} target="_blank" rel="noreferrer">
                                        {cit.url}
                                      </a>
                                    </div>
                                  )}
                                  <div style={{ marginTop: "4px", fontStyle: "italic" }}>
                                    &ldquo;{cit.excerpt}...&rdquo;
                                  </div>
                                </div>
                              )
                          )}
                        </div>
                      )}
                  </div>
                ))}

                {isStreaming && (
                  <div className="msg-card msg-assistant" aria-live="polite">
                    {streamMeta && (
                      <div className="msg-meta-bar">
                        <span className="meta-tag perm-strict">
                          Mode:{" "}
                          {PERMISSION_LABELS[streamMeta.permission_level] ||
                            streamMeta.permission_level}
                        </span>
                        <span className="meta-tag">
                          Book Coverage: {streamMeta.book_coverage}
                        </span>
                      </div>
                    )}
                    {streamingText ? (
                      <FormattedMarkdown text={streamingText} />
                    ) : (
                      <div>{requestNotice || "Preparing response…"}</div>
                    )}
                  </div>
                )}

                {requestNotice && (
                  <div role="status" style={{ padding: "8px", color: "var(--text-secondary)" }}>{requestNotice}</div>
                )}

                {chatError && (
                  <div
                    className="card-panel"
                    style={{
                      background: "var(--danger-bg)",
                      borderColor: "#fecdca",
                      color: "var(--danger)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: "12px",
                    }}
                  >
                    <div>
                      <strong>Error:</strong> {chatError}
                    </div>
                    {lastFailedPrompt && (
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ whiteSpace: "nowrap" }}
                        onClick={() => handleSendMessage(null, lastFailedPrompt)}
                      >
                        Retry Message
                      </button>
                    )}
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="chat-input-bar">
                <form className="chat-form" onSubmit={handleSendMessage}>
                  <textarea
                    className="chat-textarea"
                    placeholder="Ask a real-world question, describe an idea for your world, or request feedback..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage(e);
                      }
                    }}
                  />
                  {isPostprocessingStage(requestStage) ? (
                    <>
                      <button type="button" className="btn btn-secondary" disabled>
                        {requestStage === "postprocessing" ? (requestNotice || "Updating World Bible…") : requestStage === "committing" ? "Finishing updates…" : "Saving response…"}
                      </button>
                      <button type="button" className="btn btn-danger" onClick={handleStopGeneration}>Cancel update</button>
                    </>
                  ) : isStreaming ? (
                    <button
                      type="button"
                      className="btn btn-danger"
                      onClick={handleStopGeneration}
                      disabled={requestStage === "cancelling"}
                    >
                      {requestStage === "cancelling" ? "Cancelling…" : "Stop"}
                    </button>
                  ) : (
                    <button
                      type="submit"
                      className="btn btn-primary"
                      disabled={!chatInput.trim()}
                    >
                      Send
                    </button>
                  )}
                </form>
              </div>
            </div>
          )}

          {/* TAB 2: WORLD BIBLE WORKSPACE */}
          {activeTab === "bible" && (
            <div className="bible-layout">
              <div className="bible-toc">
                <div style={{ fontWeight: 700, marginBottom: "8px" }}>
                  World Bible Sections
                </div>
                <input
                  type="search"
                  className="input-field"
                  placeholder="Search World Bible..."
                  style={{ marginBottom: "10px" }}
                  value={sectionSearch}
                  onChange={(e) => setSectionSearch(e.target.value)}
                />
                {filteredSections.map((sec) => (
                  <div
                    key={sec.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "7px 9px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      background:
                        sec.id === activeSectionId ? "var(--accent-primary)" : "transparent",
                      color: sec.id === activeSectionId ? "#ffffff" : "var(--text-primary)",
                      fontSize: "0.86rem",
                      marginBottom: "2px",
                    }}
                    onClick={() => selectBibleSection(sec)}
                  >
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                      {sec.title}
                    </span>
                    <span style={{ fontSize: "0.7rem", opacity: 0.8 }}>
                      {(sec.markdown_content || "").trim() ? `v${sec.version_number}` : "empty"}
                    </span>
                  </div>
                ))}

                <form onSubmit={handleCreateSection} style={{ marginTop: "14px" }}>
                  <input
                    className="input-field"
                    placeholder="New custom section title..."
                    value={newSectionTitle}
                    onChange={(e) => setNewSectionTitle(e.target.value)}
                    style={{ marginBottom: "6px" }}
                  />
                  <button type="submit" className="btn btn-secondary" style={{ width: "100%" }}>
                    + Add Section
                  </button>
                </form>
              </div>

              <div className="bible-editor-pane">
                {activeSectionId ? (
                  <>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: "12px",
                        gap: "12px",
                      }}
                    >
                      <input
                        className="input-field"
                        style={{
                          fontFamily: "var(--font-serif)",
                          fontSize: "1.35rem",
                          fontWeight: 700,
                          maxWidth: "460px",
                        }}
                        value={editorTitle}
                        onChange={(e) => setEditorTitle(e.target.value)}
                      />
                      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                        { (() => {
                          const cur = sections.find((s) => s.id === activeSectionId);
                          return cur ? (
                            <span
                              className={`provenance-badge ${
                                cur.dominant_provenance === "AUTHOR_CREATED"
                                  ? "prov-author"
                                  : "prov-ai"
                              }`}
                            >
                              Provenance:{" "}
                              {cur.dominant_provenance === "AUTHOR_CREATED"
                                ? "Author Created"
                                : cur.dominant_provenance === "MIXED"
                                ? "Author + Approved AI"
                                : "Author-Approved AI"}
                            </span>
                          ) : null;
                        })() }
                        <div
                          style={{
                            display: "inline-flex",
                            border: "1px solid var(--border-subtle)",
                            borderRadius: "6px",
                            overflow: "hidden",
                          }}
                        >
                          {["edit", "split", "preview"].map((mode) => (
                            <button
                              key={mode}
                              type="button"
                              onClick={() => setBibleViewMode(mode)}
                              style={{
                                padding: "5px 10px",
                                fontSize: "0.76rem",
                                fontWeight: 600,
                                border: "none",
                                cursor: "pointer",
                                background:
                                  bibleViewMode === mode
                                    ? "var(--accent-primary)"
                                    : "var(--bg-elevated)",
                                color:
                                  bibleViewMode === mode ? "#ffffff" : "var(--text-secondary)",
                              }}
                            >
                              {mode === "edit"
                                ? "Edit"
                                : mode === "split"
                                ? "Split"
                                : "Preview"}
                            </button>
                          ))}
                        </div>
                        <button className="btn btn-primary" onClick={handleSaveBibleSection}>
                          Save Section
                        </button>
                        <button className="btn btn-secondary" onClick={handleLoadRevisions}>
                          Revision History
                        </button>
                        <button
                          className="btn btn-danger"
                          onClick={() => handleDeleteSection(activeSectionId)}
                        >
                          Delete
                        </button>
                      </div>
                    </div>

                    {saveMessage && (
                      <div
                        style={{
                          fontSize: "0.83rem",
                          color: "var(--accent-primary)",
                          marginBottom: "8px",
                          fontWeight: 600,
                        }}
                      >
                        {saveMessage} — Direct edits immediately update the agent&apos;s canon
                        understanding.
                      </div>
                    )}

                    {bibleViewMode === "edit" && (
                      <textarea
                        className="bible-textarea"
                        placeholder="Write or edit established canon for this section in Markdown. Everything written directly here is stored with Tier-1 Author Provenance..."
                        value={editorMarkdown}
                        onChange={(e) => setEditorMarkdown(e.target.value)}
                      />
                    )}

                    {bibleViewMode === "preview" && (
                      <div className="bible-preview-box">
                        <FormattedMarkdown text={editorMarkdown} />
                      </div>
                    )}

                    {bibleViewMode === "split" && (
                      <div className="bible-split-grid">
                        <textarea
                          className="bible-textarea"
                          placeholder="Write or edit established canon for this section in Markdown..."
                          value={editorMarkdown}
                          onChange={(e) => setEditorMarkdown(e.target.value)}
                        />
                        <div className="bible-preview-box">
                          <FormattedMarkdown text={editorMarkdown} />
                        </div>
                      </div>
                    )}

                    {showRevisions && (
                      <div className="card-panel" style={{ marginTop: "16px" }}>
                        <h4>Section Revision History</h4>
                        {revisions.map((r) => (
                          <div
                            key={r.id}
                            style={{
                              borderBottom: "1px solid var(--border-subtle)",
                              padding: "8px 0",
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              fontSize: "0.84rem",
                            }}
                          >
                            <div>
                              <strong>v{r.version_number}</strong> ({r.edit_origin}) —{" "}
                              {r.author_note} ({new Date(r.created_at).toLocaleString()})
                            </div>
                            <button
                              className="btn btn-secondary"
                              onClick={() => handleRevertRevision(r.version_number)}
                            >
                              Restore v{r.version_number}
                            </button>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Established Atomic Facts & Provenance Inspector */}
                    <div className="card-panel" style={{ marginTop: "20px" }}>
                      <h4 style={{ marginBottom: "8px" }}>
                        Established World Canon & Provenance Ledger ({facts.length} facts)
                      </h4>
                      {consistencyReport?.flagged_facts?.length > 0 && (
                        <div
                          style={{
                            background: "var(--accent-amber-bg)",
                            border: "1px solid #f3d19e",
                            padding: "10px",
                            borderRadius: "6px",
                            marginBottom: "10px",
                            fontSize: "0.84rem",
                          }}
                        >
                          <strong>Cross-Domain Consistency Review Needed:</strong>
                          {consistencyReport.flagged_facts.map((ff) => (
                            <div
                              key={ff.id}
                              style={{
                                marginTop: "6px",
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                gap: "10px",
                              }}
                            >
                              <div>
                                • <strong>{ff.title}:</strong> {ff.consistency_notes}
                              </div>
                              <button
                                type="button"
                                className="btn btn-secondary"
                                style={{ padding: "3px 9px", fontSize: "0.75rem", whiteSpace: "nowrap" }}
                                onClick={() => handleResolveConsistencyFlag(ff.id)}
                              >
                                Mark Reviewed
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                      {facts
                        .filter((f) => f.canon_status === "ESTABLISHED_CANON")
                        .slice(0, 15)
                        .map((f) => (
                          <div
                            key={f.id}
                            style={{
                              padding: "8px 0",
                              borderBottom: "1px solid var(--border-subtle)",
                              fontSize: "0.85rem",
                            }}
                          >
                            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                              <strong>{f.title}</strong>
                              <span
                                className={`provenance-badge ${
                                  f.provenance_type === "AUTHOR_CREATED"
                                    ? "prov-author"
                                    : "prov-ai"
                                }`}
                              >
                                {f.provenance_type === "AUTHOR_CREATED"
                                  ? "Author Created (Tier 1)"
                                  : "AI Suggested / Author Approved (Tier 2)"}
                              </span>
                            </div>
                            <div style={{ color: "var(--text-secondary)", marginTop: "2px" }}>
                              {f.content}
                            </div>
                          </div>
                        ))}
                    </div>
                  </>
                ) : (
                  <p>Select or create a World Bible section on the left.</p>
                )}
              </div>
            </div>
          )}

          {/* TAB 3: SOURCES & RESEARCH INTERFACE */}
          {activeTab === "sources" && (
            <div style={{ padding: "26px", overflowY: "auto", width: "100%" }}>
              <div className="card-panel">
                <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                  Primary Reference Book (Tier-2 Real-World Authority)
                </h3>
                <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>
                  Reference books provide real-world subject knowledge via Retrieval-Augmented
                  Generation (RAG). They never dictate or overwrite your fictional canon.
                </p>
                <div style={{ marginTop: "12px" }}>
                  {systemStatus?.reference_documents?.map((doc) => (
                    <div
                      key={doc.id}
                      style={{
                        padding: "12px",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: "6px",
                        marginBottom: "8px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <div>
                        <strong>{doc.title}</strong> by {doc.author} ({doc.file_path})
                        <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                          Status: <strong>{doc.ingestion_status}</strong> | Extracted Pages:{" "}
                          {doc.total_pages} | Semantic Passages Indexed: {doc.total_chunks} |
                          Retrieval: local full-text search
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: "12px" }}>
                  <label className="btn btn-secondary" style={{ display: "inline-block" }}>
                    {uploadingBook ? "Processing Book..." : "+ Upload Additional Reference Book (PDF/TXT/MD)"}
                    <input
                      type="file"
                      accept=".pdf,.txt,.md"
                      style={{ display: "none" }}
                      onChange={handleBookUpload}
                    />
                  </label>
                </div>
              </div>

              <div className="card-panel">
                <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                  Supplementary Online Research via Tavily (Tier-3 Real-World Authority)
                </h3>
                <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>
                  Triggered automatically when a topic is outside the scope of{" "}
                  <em>The Planet Construction Kit</em> (such as constructed languages).
                </p>
                {systemStatus?.online_sources?.length === 0 ? (
                  <p style={{ marginTop: "10px", fontSize: "0.86rem", color: "var(--text-muted)" }}>
                    No online sources retrieved yet. Ask a specialized research question in chat to
                    populate verified web citations.
                  </p>
                ) : (
                  systemStatus?.online_sources?.map((s) => (
                    <div
                      key={s.id}
                      style={{
                        borderTop: "1px solid var(--border-subtle)",
                        padding: "10px 0",
                        marginTop: "8px",
                        fontSize: "0.86rem",
                      }}
                    >
                      <div>
                        <strong>{s.title}</strong> ({s.domain}) —{" "}
                        <span className="badge-pill" style={{ background: "var(--bg-subtle)" }}>
                          {s.credibility_tier}
                        </span>
                      </div>
                      <div>
                        <a href={s.url} target="_blank" rel="noreferrer">
                          {s.url}
                        </a>
                      </div>
                      <div style={{ color: "var(--text-secondary)", marginTop: "4px" }}>
                        {s.excerpt}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* TAB 4: PROJECT & APPLICATION SETTINGS */}
          {activeTab === "settings" && (
            <div style={{ padding: "26px", overflowY: "auto", width: "100%" }}>
              <div className="card-panel">
                <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                  Application & Local Model Configuration
                </h3>
                <p style={{ fontSize: "0.86rem", color: "var(--text-secondary)" }}>
                  Run Ollama locally and download a model before selecting it here. No model API key
                  is required. Model selection is saved locally. Tavily web research is optional;
                  leave its key empty to work offline.
                </p>
                <form
                  onSubmit={handleUpdateModel}
                  style={{ display: "flex", gap: "10px", marginTop: "12px", maxWidth: "540px", alignItems: "center" }}
                >
                  <input
                    className="input-field"
                    value={modelInput}
                    onChange={(e) => setModelInput(e.target.value)}
                    placeholder="e.g. gpt-oss:20b"
                  />
                  <button type="submit" className="btn btn-primary" style={{ whiteSpace: "nowrap" }}>
                    Update Model
                  </button>
                  {modelSaveMsg && (
                    <span style={{ fontSize: "0.84rem", color: "var(--accent-primary)", fontWeight: 600 }}>
                      {modelSaveMsg}
                    </span>
                  )}
                </form>
              </div>

              {activeWorldDetails && (
                <>
                  <div className="card-panel">
                    <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                      World Name & Overview
                    </h3>
                    <form
                      onSubmit={handleUpdateWorldDetails}
                      style={{ display: "flex", flexDirection: "column", gap: "10px", maxWidth: "520px" }}
                    >
                      <div>
                        <label style={{ fontSize: "0.82rem", fontWeight: 600 }}>World Name</label>
                        <input
                          className="input-field"
                          required
                          value={editWorldName}
                          onChange={(e) => setEditWorldName(e.target.value)}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: "0.82rem", fontWeight: 600 }}>Description</label>
                        <input
                          className="input-field"
                          value={editWorldDesc}
                          onChange={(e) => setEditWorldDesc(e.target.value)}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                          Setting / Tone Notes
                        </label>
                        <input
                          className="input-field"
                          value={editWorldGenre}
                          onChange={(e) => setEditWorldGenre(e.target.value)}
                        />
                      </div>
                      <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                        <button type="submit" className="btn btn-primary">
                          Save World Details
                        </button>
                        {worldSaveMsg && (
                          <span style={{ fontSize: "0.84rem", color: "var(--accent-primary)", fontWeight: 600 }}>
                            {worldSaveMsg}
                          </span>
                        )}
                      </div>
                    </form>
                  </div>

                  <div className="card-panel">
                    <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                      Subject-Specific Author Expertise Profile
                    </h3>
                    <p style={{ fontSize: "0.86rem", color: "var(--text-secondary)" }}>
                      The consultant adapts its explanation depth per subject through conversation.
                      You can also override any subject level manually below:
                    </p>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                        gap: "10px",
                        marginTop: "12px",
                      }}
                    >
                      {activeWorldDetails.expertise_profiles?.map((ep) => (
                        <div
                          key={ep.domain_key}
                          style={{
                            border: "1px solid var(--border-subtle)",
                            borderRadius: "6px",
                            padding: "10px",
                            fontSize: "0.84rem",
                          }}
                        >
                          <div style={{ fontWeight: 600 }}>{ep.domain_name}</div>
                          <select
                            className="input-field"
                            style={{ marginTop: "6px" }}
                            value={ep.expertise_level}
                            onChange={(e) =>
                              handleExpertiseOverride(ep.domain_key, e.target.value)
                            }
                          >
                            <option value="UNKNOWN">Auto-Infer (Unknown)</option>
                            <option value="BEGINNER">Beginner (Explain Foundations)</option>
                            <option value="INTERMEDIATE">Intermediate</option>
                            <option value="ADVANCED">Advanced</option>
                            <option value="EXPERT">Expert (Peer Technical Level)</option>
                          </select>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card-panel">
                    <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                      Project Backup & Restore Archive (.JSON)
                    </h3>
                    <p style={{ fontSize: "0.86rem", color: "var(--text-secondary)", marginBottom: "12px" }}>
                      Export a complete, portable JSON backup of <strong>{activeWorldDetails.name}</strong>{" "}
                      (including all World Bible sections, revision history, established canon facts,
                      cross-domain dependencies, and conversation threads), or restore a backup file.
                    </p>
                    <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
                      <a
                        className="btn btn-primary"
                        href={`/api/worlds/${activeWorldId}/export-project`}
                        download
                      >
                        Download Full Project Backup (.JSON)
                      </a>
                      <label className="btn btn-secondary" style={{ display: "inline-block" }}>
                        Import / Restore Project Archive (.JSON)
                        <input
                          type="file"
                          accept=".json,application/json"
                          style={{ display: "none" }}
                          onChange={handleImportProjectBundle}
                        />
                      </label>
                    </div>
                  </div>

                  <div className="card-panel">
                    <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "8px" }}>
                      Danger Zone — Delete World Project
                    </h3>
                    <button className="btn btn-danger" onClick={handleDeleteWorld}>
                      Delete &ldquo;{activeWorldDetails.name}&rdquo;
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Modal for Creating a New World */}
      {showNewWorldModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 50,
          }}
        >
          <div className="card-panel" style={{ width: "440px", maxWidth: "92vw" }}>
            <h3 style={{ fontFamily: "var(--font-serif)", marginBottom: "10px" }}>
              Create Fictional World Project
            </h3>
            <form onSubmit={handleCreateWorld}>
              <div style={{ marginBottom: "10px" }}>
                <label style={{ fontSize: "0.82rem", fontWeight: 600 }}>World Name *</label>
                <input
                  className="input-field"
                  required
                  placeholder="e.g. The Thalassian League"
                  value={newWorldName}
                  onChange={(e) => setNewWorldName(e.target.value)}
                />
              </div>
              <div style={{ marginBottom: "10px" }}>
                <label style={{ fontSize: "0.82rem", fontWeight: 600 }}>Brief Description</label>
                <input
                  className="input-field"
                  placeholder="e.g. Bronze-age volcanic archipelago"
                  value={newWorldDesc}
                  onChange={(e) => setNewWorldDesc(e.target.value)}
                />
              </div>
              <div style={{ marginBottom: "14px" }}>
                <label style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                  Setting / Physics Notes (Optional)
                </label>
                <input
                  className="input-field"
                  placeholder="e.g. Non-magical historical realism"
                  value={newWorldGenre}
                  onChange={(e) => setNewWorldGenre(e.target.value)}
                />
              </div>
              <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
                {worlds.length > 0 && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setShowNewWorldModal(false)}
                  >
                    Cancel
                  </button>
                )}
                <button type="submit" className="btn btn-primary">
                  Create World
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
