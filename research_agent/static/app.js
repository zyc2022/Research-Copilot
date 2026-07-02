const state = {
  conversationId: null,
  kbId: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).detail || detail; } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function setStatus(text) {
  $("statusPill").textContent = text;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m]));
}

function renderMessage(role, content, citations = []) {
  const el = document.createElement("article");
  el.className = `message ${role}`;
  el.innerHTML = escapeHtml(content);
  if (citations.length) {
    const list = document.createElement("div");
    list.className = "citations";
    list.innerHTML = citations.map((c, idx) =>
      `<span class="citation">[${idx + 1}] ${escapeHtml(c.kb_name)} / ${escapeHtml(c.filename)} · ${c.score}</span>`
    ).join("");
    el.appendChild(list);
  }
  $("chatMessages").appendChild(el);
  $("chatMessages").scrollTop = $("chatMessages").scrollHeight;
}

async function loadConversations() {
  const items = await api("/conversations");
  const list = $("conversationList");
  list.innerHTML = "";
  for (const item of items) {
    const row = document.createElement("button");
    row.className = `conversation-item ${item.id === state.conversationId ? "active" : ""}`;
    row.innerHTML = `<span>${escapeHtml(item.title)}</span><span class="delete" title="删除">×</span>`;
    row.addEventListener("click", async (event) => {
      if (event.target.classList.contains("delete")) {
        event.stopPropagation();
        await api(`/conversations/${item.id}`, { method: "DELETE" });
        if (state.conversationId === item.id) {
          state.conversationId = null;
          $("chatMessages").innerHTML = "";
        }
        await loadConversations();
        return;
      }
      state.conversationId = item.id;
      await loadMessages(item.id);
      await loadConversations();
    });
    list.appendChild(row);
  }
}

async function loadMessages(conversationId) {
  const messages = await api(`/conversations/${conversationId}/messages`);
  $("chatMessages").innerHTML = "";
  for (const msg of messages) {
    renderMessage(msg.role, msg.content, msg.citations || []);
  }
}

async function newConversation() {
  const conv = await api("/conversations", {
    method: "POST",
    body: JSON.stringify({ title: "新对话" }),
  });
  state.conversationId = conv.id;
  $("chatMessages").innerHTML = "";
  await loadConversations();
}

async function loadModel() {
  const cfg = await api("/model");
  $("modelBaseUrl").value = cfg.base_url || "";
  $("modelApiKey").value = cfg.api_key || "";
  $("modelName").value = cfg.model || "";
}

async function saveModel(event) {
  event.preventDefault();
  await api("/model", {
    method: "POST",
    body: JSON.stringify({
      base_url: $("modelBaseUrl").value.trim(),
      api_key: $("modelApiKey").value.trim(),
      model: $("modelName").value.trim(),
    }),
  });
  $("modelDialog").close();
  setStatus("模型配置已保存");
}

async function loadKnowledgeBases() {
  const items = await api("/knowledge-bases");
  const list = $("kbList");
  list.innerHTML = "";
  for (const kb of items) {
    const card = document.createElement("div");
    card.className = `kb-card ${kb.id === state.kbId ? "active" : ""}`;
    card.innerHTML = `
      <h4>${escapeHtml(kb.name)}</h4>
      <p>${escapeHtml(kb.description || "无描述")}</p>
      <p>Embedding: ${escapeHtml(kb.embedding_model)} · 文档 ${kb.document_count} · 片段 ${kb.chunk_count}</p>
      <div class="kb-actions">
        <button class="small toggle-kb ${kb.enabled ? "enabled" : "disabled"}">${kb.enabled ? "已启用" : "已禁用"}</button>
        <button class="small open">查看</button>
        <button class="small delete-kb">删除</button>
      </div>
    `;
    card.querySelector(".open").addEventListener("click", async () => {
      state.kbId = kb.id;
      showKbDetail(kb);
      await loadKnowledgeBases();
      await loadDocuments(kb.id);
    });
    card.querySelector(".toggle-kb").addEventListener("click", async () => {
      const nextEnabled = !Boolean(kb.enabled);
      await api(`/knowledge-bases/${kb.id}/enabled?enabled=${nextEnabled}`, { method: "POST" });
      setStatus(nextEnabled ? "知识库已启用" : "知识库已禁用");
      await loadKnowledgeBases();
    });
    card.querySelector(".delete-kb").addEventListener("click", async () => {
      if (!confirm(`删除知识库“${kb.name}”？`)) return;
      await api(`/knowledge-bases/${kb.id}`, { method: "DELETE" });
      if (state.kbId === kb.id) {
        state.kbId = null;
        $("kbDetail").classList.add("hidden");
        $("kbDetailEmpty").classList.remove("hidden");
      }
      await loadKnowledgeBases();
    });
    list.appendChild(card);
  }
}

function showKbDetail(kb) {
  $("kbDetailEmpty").classList.add("hidden");
  $("kbDetail").classList.remove("hidden");
  $("kbDetailTitle").textContent = kb.name;
  $("kbDetailMeta").textContent = `Embedding: ${kb.embedding_model}，创建后不可修改`;
}

async function loadDocuments(kbId) {
  const docs = await api(`/knowledge-bases/${kbId}/documents`);
  const list = $("documentList");
  list.innerHTML = docs.length ? "" : `<div class="empty">还没有文档，上传 TXT 或 PDF 开始索引。</div>`;
  for (const doc of docs) {
    const card = document.createElement("div");
    card.className = "doc-card";
    card.innerHTML = `
      <h4>${escapeHtml(doc.filename)}</h4>
      <p>${doc.file_type.toUpperCase()} · ${doc.chunk_count} 个片段 · ${doc.status}</p>
      <button class="small delete-doc">删除</button>
    `;
    card.querySelector(".delete-doc").addEventListener("click", async () => {
      await api(`/documents/${doc.id}`, { method: "DELETE" });
      await loadDocuments(kbId);
      await loadKnowledgeBases();
    });
    list.appendChild(card);
  }
}

async function createKb(event) {
  event.preventDefault();
  await api("/knowledge-bases", {
    method: "POST",
    body: JSON.stringify({
      name: $("kbName").value.trim(),
      description: $("kbDescription").value.trim(),
      embedding_base_url: $("embeddingBaseUrl").value.trim(),
      embedding_api_key: $("embeddingApiKey").value.trim(),
      embedding_model: $("embeddingModel").value.trim() || "local-hash",
    }),
  });
  $("createKbDialog").close();
  event.target.reset();
  $("embeddingModel").value = "local-hash";
  await loadKnowledgeBases();
}

async function uploadDocument(event) {
  const file = event.target.files[0];
  if (!file || !state.kbId) return;
  const form = new FormData();
  form.append("file", file);
  setStatus("正在解析并索引文档...");
  try {
    await api(`/knowledge-bases/${state.kbId}/documents`, { method: "POST", body: form });
    await loadDocuments(state.kbId);
    await loadKnowledgeBases();
    setStatus("文档已索引");
  } catch (err) {
    alert(err.message);
    setStatus("索引失败");
  } finally {
    event.target.value = "";
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const input = $("messageInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  renderMessage("user", message);
  setStatus("检索并生成回答中...");
  try {
    const result = await api("/chat", {
      method: "POST",
      body: JSON.stringify({ message, conversation_id: state.conversationId }),
    });
    state.conversationId = result.conversation_id;
    renderMessage("assistant", result.answer, result.citations || []);
    await loadConversations();
    setStatus("准备就绪");
  } catch (err) {
    renderMessage("assistant", `请求失败：${err.message}`);
    setStatus("请求失败");
  }
}

document.querySelectorAll("[data-close]").forEach((btn) => {
  btn.addEventListener("click", () => $(btn.dataset.close).close());
});

$("newConversation").addEventListener("click", newConversation);
$("chatForm").addEventListener("submit", sendMessage);
$("modelButton").addEventListener("click", async () => {
  await loadModel();
  $("modelDialog").showModal();
});
$("modelForm").addEventListener("submit", saveModel);
$("kbButton").addEventListener("click", async () => {
  await loadKnowledgeBases();
  $("kbDialog").showModal();
});
$("showCreateKb").addEventListener("click", () => $("createKbDialog").showModal());
$("createKbForm").addEventListener("submit", createKb);
$("documentUpload").addEventListener("change", uploadDocument);

newConversation();
