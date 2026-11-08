import React, { useState, useEffect, useRef } from "react";

const PERMISSION_LABELS = {
  FACTUAL_ONLY: "Factual Explanation Only (No Fictional Invention)",
  ANALYTICAL_ADVISORY: "Analytical & Consequence Review",
  IMPLEMENTATION_SCOPED: "Implementation Assistance (Author Premise)",
  BOUNDED_BRAINSTORMING: "Opt-In Scoped Brainstorming",
  EXPLICIT_DRAFTING: "Author-Authorized World Bible Drafting",
};

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
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [streamMeta, setStreamMeta] = useState(null);
  const [chatError, setChatError] = useState("");
  const [expandedCitation, setExpandedCitation] = useState(null);
  const abortControllerRef = useRef(null);
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
  const [modelInput, setModelInput] = useState("gemini-2.5-flash");

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
      if (data.gemini_model) setModelInput(data.gemini_model);
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

      setActiveWorldDetails(details);
      setSections(secs);
      setProposals(props);
      setFacts(factsData);
      setConsistencyReport(consData);

      if (secs.length > 0) {
        selectBibleSection(secs[0]);
      } else {
        setActiveSectionId("");
        setEditorTitle("");
        setEditorMarkdown("");
      }

      setConversations(convs);
      if (convs.length > 0) {
        setActiveConvId(convs[0].id);
      } else {
        setActiveConvId("");
        setMessages([]);
      }
    } catch (err) {
      console.error("Error loading world workspace:", err);
    }
  }

  async function fetchMessages(convId) {
    try {
      const res = await fetch(`/api/conversations/${convId}/messages`);
      const data = await res.json();
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

  async function handleSendMessage(e) {
    e.preventDefault();
    if (!chatInput.trim() || !activeConvId || isStreaming) return;
    const promptText = chatInput.trim();
    setChatInput("");
    setChatError("");
    setIsStreaming(true);
    setStreamingText("");
    setStreamMeta(null);

    const optimisticUserMsg = {
      id: "temp-user-" + Date.now(),
      role: "USER",
      content: promptText,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMsg]);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const res = await fetch(`/api/conversations/${activeConvId}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: promptText }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const msg =
          errData?.detail?.message ||
          errData?.detail ||
          `Request failed with status ${res.status}`;
        throw new Error(msg);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const ev of events) {
          const line = ev.trim();
          if (!line.startsWith("data: ")) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.type === "turn_meta") {
            setStreamMeta(payload);
          } else if (payload.type === "delta") {
            setStreamingText((prev) => prev + payload.text);
          } else if (payload.type === "complete") {
            setMessages((prev) => [...prev, payload.assistant_message]);
            setStreamingText("");
            setStreamMeta(null);
            // Refresh proposals and expertise profile
            loadWorldWorkspace(activeWorldId);
            fetchSystemStatus();
          } else if (payload.type === "error") {
            throw new Error(payload.message || "Streaming error");
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setChatError(err.message || "Failed communicating with the Gemini API.");
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }

  function handleStopGeneration() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
    }
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

  async function handleUpdateModel(e) {
    e.preventDefault();
    await fetch("/api/system/model", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gemini_model: modelInput }),
    });
    await fetchSystemStatus();
  }

  const filteredSections = sections.filter(
    (s) =>
      s.title.toLowerCase().includes(sectionSearch.toLowerCase()) ||
      (s.markdown_content || "").toLowerCase().includes(sectionSearch.toLowerCase())
  );

  return (
    <div className="app-shell">
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
          style={{ width: "100%", marginBottom: "10px", fontSize: "0.8rem" }}
          onClick={() => setShowNewWorldModal(true)}
        >
          + New Fictional World
        </button>

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
                systemStatus?.has_valid_claude_key ? "status-ok" : "status-warn"
              }`}
            />
            Claude API:{" "}
            {systemStatus?.has_valid_claude_key
              ? `Ready (${systemStatus.claude_model || "claude-sonnet-5-5"})`
              : "Add Key in .env"}
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
            <span className="topbar-world-name">
              {activeWorldDetails?.name || "No World Selected"}
            </span>
            {activeWorldDetails?.description && (
              <span style={{ fontSize: "0.84rem", color: "var(--text-secondary)" }}>
                — {activeWorldDetails.description}
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
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
              </>
            )}
          </div>
        </header>

        {/* Skippable First-Run Onboarding Banner */}
        {showOnboarding && (
          <div
            style={{
              background: "#eef5fb",
              borderBottom: "1px solid #cbe0f5",
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
                    <ul
                      style={{
                        marginTop: "10px",
                        marginLeft: "20px",
                        fontSize: "0.88rem",
                        color: "var(--text-secondary)",
                      }}
                    >
                      <li>
                        <strong>Ask how real-world systems work:</strong>{" "}
                        <em>&ldquo;How did medieval taxation work before standing bureaucracies?&rdquo;</em>
                      </li>
                      <li>
                        <strong>Implement your own creative concept:</strong>{" "}
                        <em>
                          &ldquo;I want a government where people elect their leaders, but only
                          landowners who completed naval service can vote.&rdquo;
                        </em>
                      </li>
                      <li>
                        <strong>Request opt-in brainstorming only when you want options:</strong>{" "}
                        <em>&ldquo;Give me three possible ways this council could handle succession.&rdquo;</em>
                      </li>
                    </ul>
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
                        {m.metadata.book_coverage && (
                          <span className="meta-tag">
                            Book Coverage: {m.metadata.book_coverage}
                          </span>
                        )}
                      </div>
                    )}

                    <div>{m.content}</div>

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
                                    background: "#faf8f4",
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
                  <div className="msg-card msg-assistant">
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
                    <div>{streamingText || "Consulting reference sources..."}</div>
                  </div>
                )}

                {chatError && (
                  <div
                    className="card-panel"
                    style={{
                      background: "var(--danger-bg)",
                      borderColor: "#fecdca",
                      color: "var(--danger)",
                    }}
                  >
                    <strong>Error:</strong> {chatError}
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
                  {isStreaming ? (
                    <button
                      type="button"
                      className="btn btn-danger"
                      onClick={handleStopGeneration}
                    >
                      Stop
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

                    <textarea
                      className="bible-textarea"
                      placeholder="Write or edit established canon for this section in Markdown. Everything written directly here is stored with Tier-1 Author Provenance..."
                      value={editorMarkdown}
                      onChange={(e) => setEditorMarkdown(e.target.value)}
                    />

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
                            <div key={ff.id} style={{ marginTop: "4px" }}>
                              • <strong>{ff.title}:</strong> {ff.consistency_notes}
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
                          Vector Embeddings: {doc.embeddings_count}
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
                        <span className="badge-pill" style={{ background: "#eee" }}>
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
                  Application & Gemini Model Configuration
                </h3>
                <p style={{ fontSize: "0.86rem", color: "var(--text-secondary)" }}>
                  API keys are loaded securely from your local <code>.env</code> file and are never
                  exposed to the browser.
                </p>
                <form
                  onSubmit={handleUpdateModel}
                  style={{ display: "flex", gap: "10px", marginTop: "12px", maxWidth: "480px" }}
                >
                  <input
                    className="input-field"
                    value={modelInput}
                    onChange={(e) => setModelInput(e.target.value)}
                    placeholder="e.g. gemini-2.5-flash"
                  />
                  <button type="submit" className="btn btn-primary">
                    Update Model
                  </button>
                </form>
              </div>

              {activeWorldDetails && (
                <>
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
