// Node 内置 runner；隔离 DOM/网络，不访问真实录音或服务。
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
    constructor() {
        this.value = ''; this.innerHTML = ''; this.textContent = '';
        this.style = {}; this.dataset = {}; this.children = [];
        this.classList = { add() {}, remove() {}, toggle() {} };
    }
    set innerHTML(value) { this._html = value; this.children = []; }
    get innerHTML() { return this._html; }
    addEventListener() {}
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    querySelector(query) {
        const id = query.match(/data-segment-id="([^"]+)"/);
        return id ? this.children.find(c => c.dataset.segmentId === id[1]) || null : null;
    }
    querySelectorAll() { return this.children; }
    focus() {}
}

function client({ storage = new Map(), fetch = () => { throw new Error('Network is forbidden in unit tests'); } } = {}) {
    const elements = new Map();
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, new Element());
            return elements.get(id);
        },
        createElement() { return new Element(); },
        addEventListener() {}, querySelector() { return new Element(); }, querySelectorAll() { return []; },
    };
    const scope = vm.createContext({ document, console, URL, URLSearchParams, TextDecoder,
        window: { location: { origin: '' } },
        alert() {}, confirm() { return false; },
        fetch, crypto: require('node:crypto').webcrypto,
        localStorage: { getItem: k => storage.get(k) ?? null, setItem: (k,v) => storage.set(k,v), removeItem: k => storage.delete(k) },
        setTimeout, clearTimeout });
    const html = fs.readFileSync(path.join(__dirname, '../static/web_client.html'), 'utf8');
    const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
    vm.runInContext(script, scope);
    return { scope, elements, storage, get: code => vm.runInContext(code, scope) };
}

function sse(events) {
    const bytes = new TextEncoder().encode(events.map(e => 'data: ' + JSON.stringify(e) + '\n\n').join(''));
    return new Response(new ReadableStream({ start(controller) {
        // 故意拆开 UTF-8 字符和 SSE 边界。
        for (let i=0; i<bytes.length; i++) controller.enqueue(bytes.slice(i, i+1));
        controller.close();
    }}));
}

const row = { time:'00:01', start_ms:1000, end_ms:2000, speaker:'未知', confidence:0, text:'测试 <img src=x>' };

test('real-time revisions replace rather than duplicate and escape output', () => {
    const c = client();
    c.scope.addTranscript({ ...row, segmentId:'s1', revision:1, isFinal:false });
    c.scope.addTranscript({ ...row, segmentId:'s1', revision:2, isFinal:true, text:'句末 <img src=x>' });
    assert.equal(c.get('liveTranscriptItems.length'), 1);
    assert.equal(c.elements.get('transcriptBox').children.length, 1);
    const rendered = c.elements.get('transcriptBox').children[0].innerHTML;
    assert.ok(rendered.includes('&lt;img src=x&gt;'));
    assert.ok(!rendered.includes('<img'));
});

test('offline review uses independent state and requires done', async () => {
    const c = client();
    c.scope.addTranscript({ ...row, segmentId:'s1', revision:1, isFinal:true });
    const before = c.get('JSON.stringify(liveTranscriptItems)');
    const response = sse([{ type:'segment', ...row }, { type:'done', segments:1 }]);
    await c.scope.consumeMeetingSseStream(response, 'test.wav');
    assert.equal(c.get('meetingTranscriptItems.length'), 1);
    assert.equal(c.get('JSON.stringify(liveTranscriptItems)'), before);
    assert.equal(c.elements.get('btnUploadSummary').disabled, false);
    assert.equal(response.body.locked, false);
    assert.ok(c.elements.get('meetingTranscript').children[0].innerHTML.includes('&lt;img'));
});

test('offline identity distinguishes rejected scores, no sampling, conflicts and confirmed matches', async () => {
    const cases = [
        [{ identityStatus:'unconfirmed', voiceprintScore:.2487 }, '未通过声纹验证；最高匹配分数: 0.2487'],
        [{ identityStatus:'not_requested', voiceprintScore:null }, '未选择参会人，仅匿名分人'],
        [{ identityStatus:'insufficient_audio', voiceprintScore:null }, '有效音频不足'],
        [{ identityStatus:'unavailable', voiceprintScore:null }, '声纹验证失败'],
        [{ identityStatus:'conflicting_windows', voiceprintScore:.8 }, '窗口身份冲突；最高匹配分数: 0.8000'],
        [{ speakerId:'a', identityStatus:'matched', confidence:.32 }, '声纹分数: 0.32'],
        [{}, '未确认身份：未通过声纹验证'],
    ];
    for (const [extra, expected] of cases) {
        const c = client();
        await c.scope.consumeMeetingSseStream(sse([{ type:'segment', ...row, ...extra }, { type:'done', segments:1 }]), 'test.wav');
        const html = c.elements.get('meetingTranscript').children[0].innerHTML;
        assert.ok(html.includes(expected), html);
        if (!extra.speakerId) assert.ok(!html.includes('(声纹分数: 0)'));
    }
});

test('SSE EOF without done or explicit error cannot be accepted as complete', async () => {
    for (const events of [
        [{ type:'segment', ...row }],
        [{ type:'error', message:'模型失败' }],
        [{ type:'segment', ...row }, { type:'done', segments:2 }],
        [{ type:'segment', ...row, index:3 }, { type:'done', segments:1 }],
    ]) {
        const c = client();
        c.scope.resetMeetingResultUi();
        const response = sse(events);
        await assert.rejects(c.scope.consumeMeetingSseStream(response, 'test.wav'));
        assert.equal(c.elements.get('btnUploadSummary').disabled, true);
        assert.equal(response.body.locked, false);
    }
    const c = client();
    await assert.rejects(c.scope.consumeMeetingSseStream(new Response('data: malformed\n\ndata: {"type":"done","segments":0}\n\n'), 'test.wav'));
});

test('saved recording remains bound to its original server and candidate scope', () => {
    const c = client();
    c.get("liveSessionServerUrl = 'http://original'; liveSessionParticipants = ['speaker-a'];");
    c.scope.renderLiveRecordingFile('test-file-id');
    c.get("serverUrlInput.value = 'http://other'; liveSessionParticipants = ['speaker-b'];");
    let called;
    c.scope.transcribeLiveRecording = (...args) => { called = args; };
    const actions = c.elements.get('liveRecordingFile').children[1];
    assert.equal(actions.children[0].href, 'http://original/v1/meeting/recordings/test-file-id');
    actions.children[1].onclick();
    assert.deepEqual(JSON.parse(JSON.stringify(called)), ['test-file-id', 'http://original', ['speaker-a']]);
});

test('renders real segments before EOF while exports remain disabled until done', async () => {
    const c = client();
    c.scope.resetMeetingResultUi();
    let controller;
    const response = new Response(new ReadableStream({ start(value) { controller = value; } }));
    const receiving = c.scope.consumeMeetingSseStream(response, 'test.wav');
    const send = event => controller.enqueue(new TextEncoder().encode('data: ' + JSON.stringify(event) + '\n\n'));
    send({ type:'segment', index:0, ...row });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(c.get('meetingTranscriptItems.length'), 1);
    assert.equal(c.elements.get('meetingTranscript').children.length, 1);
    assert.equal(c.elements.get('btnUploadSummary').disabled, true);
    assert.equal(c.elements.get('btnDownloadMeeting').disabled, true);
    send({ type:'info', method:'moss', total_segments:1 });
    send({ type:'done', segments:1 });
    controller.close();
    await receiving;
    assert.equal(c.elements.get('btnDownloadMeeting').disabled, false);
    assert.equal(c.elements.get('btnUploadSummary').disabled, false);
});

test('partial failure remains visible without enabling export or summary', () => {
    const c = client();
    c.scope.resetMeetingResultUi();
    c.scope.showMeetingFailure(new Error('中断 <img src=x>'));
    assert.equal(c.elements.get('uploadStatus').style.display, 'block');
    assert.ok(c.elements.get('uploadStatus').innerHTML.includes('转写未完成'));
    assert.ok(c.elements.get('uploadStatus').innerHTML.includes('&lt;img'));
    assert.equal(c.elements.get('btnUploadSummary').disabled, true);
    assert.equal(c.elements.get('btnDownloadMeeting').disabled, true);
});

test('a refreshed or reopened page replays the same job from its original server without any POST', async () => {
    const first = client();
    const id = first.scope.rememberMeetingJob('https://original.example', 'private-name.wav');
    assert.ok(![...first.storage.values()].join('').includes('private-name'));
    const calls = [];
    const refreshed = client({ storage: first.storage, fetch: async (url, options) => {
        calls.push([url, options?.method || 'GET']);
        if (url.endsWith('/events?after=0')) return sse([
            { type:'status', eventId:1, jobId:id, message:'恢复中' },
            { type:'segment', eventId:2, index:0, ...row },
            { type:'done', eventId:3, segments:1 },
        ]);
        return Response.json({ jobId:id, sourceLabel:'restored.wav', state:'running' });
    }});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(refreshed.get('meetingTranscriptItems.length'), 1);
    assert.equal(refreshed.get('meetingRequestInProgress'), false);
    assert.equal(refreshed.get('activeMeetingJob.terminal'), true);
    assert.ok(calls.length === 2 && calls.every(([url, method]) => method === 'GET' && url.startsWith('https://original.example/')));
    assert.ok(![...first.storage.values()].join('').includes('测试'));
});

test('temporary disconnect reconnects from the last applied event without duplicate text', async () => {
    const c = client();
    c.scope.resetMeetingResultUi();
    const id = c.scope.rememberMeetingJob('https://original.example', 'test.wav');
    c.scope.setTimeout = fn => setTimeout(fn, 0);
    const calls = [];
    c.scope.fetch = async (url, options) => {
        calls.push([url, options?.method || 'GET']);
        return sse([{ type:'done', eventId:3, segments:1 }]);
    };
    let controller;
    const response = new Response(new ReadableStream({ start(value) { controller = value; } }));
    const following = c.scope.followMeetingJob(response);
    for (const event of [ { type:'status', eventId:1, jobId:id, message:'处理中' }, { type:'segment', eventId:2, index:0, ...row } ]) {
        controller.enqueue(new TextEncoder().encode('data: ' + JSON.stringify(event) + '\n\n'));
    }
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(c.get('meetingTranscriptItems.length'), 1);
    controller.error(new Error('network disconnected'));
    await following;
    assert.equal(c.get('meetingTranscriptItems.length'), 1);
    assert.equal(c.elements.get('meetingTranscript').children.length, 1);
    assert.deepEqual(calls, [[`https://original.example/v1/meeting/jobs/${id}/events?after=2`, 'GET']]);
});

test('a terminal job error is not retried as a new inference', async () => {
    const c = client();
    const id = c.scope.rememberMeetingJob('https://original.example', 'test.wav');
    let calls = 0;
    c.scope.fetch = async () => { calls++; throw new Error('unexpected request'); };
    await assert.rejects(c.scope.followMeetingJob(sse([
        { type:'status', eventId:1, jobId:id, message:'处理中' },
        { type:'error', eventId:2, message:'已取消', cancelled:true },
    ])), /已取消/);
    assert.equal(calls, 0);
    assert.equal(c.get('activeMeetingJob.terminal'), true);
});

test('cancel racing with done does not bring back the loading indicator', async () => {
    const c = client();
    c.scope.rememberMeetingJob('https://original.example', 'test.wav');
    c.scope.confirm = () => true;
    let finish;
    c.scope.fetch = () => new Promise(resolve => { finish = resolve; });
    const cancelling = c.scope.cancelMeetingJob();
    c.get('activeMeetingJob.terminal = true');
    c.get("document.getElementById('uploadStatus').style.display = 'none'");
    finish(Response.json({ state:'done' }));
    await cancelling;
    assert.equal(c.elements.get('uploadStatus').style.display, 'none');
});

test('the removed explanatory copy does not return', () => {
    const html = fs.readFileSync(path.join(__dirname, '../static/web_client.html'), 'utf8');
    assert.ok(!html.includes('实时稿：边说边显示，句末修正。'));
    assert.ok(!html.includes('保留完整上下文与匿名分人'));
});

test('summary requires confirmation and renders model text as data, not HTML', async () => {
    const c = client();
    c.scope.addTranscript({ ...row, segmentId:'s1', revision:1, isFinal:true });
    let calls = 0;
    c.scope.fetch = async () => { calls++; };
    await c.scope.requestSummary('live');
    assert.equal(calls, 0);
    const container = new Element();
    c.scope.renderSummary(container, '**待核对** <img src=x>');
    assert.ok(container.innerHTML.includes('<strong>待核对</strong>'));
    assert.ok(container.innerHTML.includes('&lt;img src=x&gt;'));
    assert.ok(!container.innerHTML.includes('<img'));
});

test('recording-only admission shows a notice instead of transcripts and ignores other status phases', () => {
    const c = client();
    c.scope.handleLiveStatus({ type:'status', phase:'fallback', message:'流式首遍识别失败，继续句末精修' });
    assert.equal(c.get('liveRecordingOnly'), false);
    assert.equal(c.get('liveRecordingStatusText()'), '录音中...');
    c.scope.handleLiveStatus({ type:'status', phase:'recording_only', message:'<b>忙</b>' });
    assert.equal(c.get('liveRecordingOnly'), true);
    assert.match(c.elements.get('statusText').textContent, /仅录音/);
    const notice = c.elements.get('transcriptBox').innerHTML;
    assert.match(notice, /MOSS 会后复核/);
    assert.ok(!notice.includes('<b>'));
    assert.equal(c.get('liveTranscriptItems.length'), 0);
});

const kwsDoc = {
    enabled: true, version: 3, updated_at: '2026-10-30T14:02:00+08:00', active_sessions: 2,
    categories: [
        { code: 'bed', name: '床号', slot: true, keywords: [{ phrase: '十二床', value: '12床', boost: null, threshold: null, tokens: 'sh í èr ch uáng' }] },
        { code: 'command', name: '查房指令', slot: false, keywords: [{ phrase: '下一床', value: 'next_bed', boost: null, threshold: 0.3, tokens: 'x ià y ī ch uáng' }] },
    ],
};

function kwsClient(handlers) {
    const calls = [];
    const c = client({ fetch: async (url, options = {}) => {
        const method = options.method || 'GET';
        calls.push({ url, method, body: options.body && JSON.parse(options.body) });
        const [status, body] = handlers[`${method} ${new URL(url, 'http://x').pathname}`](options.body && JSON.parse(options.body));
        return new Response(JSON.stringify(body), { status });
    } });
    return { ...c, calls };
}

const settle = () => new Promise(resolve => setTimeout(resolve, 0));
const key = (k, value = '') => ({ key: k, target: { value }, preventDefault() {}, stopPropagation() {} });

test('kws spoken numbers turn digits into Chinese phrases', () => {
    const c = client();
    assert.deepEqual([1, 10, 12, 20, 99, 100, 105, 110, 999].map(n => c.scope.kwsToChinese(n)),
        ['一', '十', '十二', '二十', '九十九', '一百', '一百零五', '一百一十', '九百九十九']);
    assert.equal(c.scope.kwsSpeak('12床'), '十二床');
    assert.equal(c.scope.kwsSpeak('1000床'), '1000床');
    assert.equal(c.scope.kwsDefaultValue('十五床', '十五床', true), '15床');
    assert.equal(c.scope.kwsDefaultValue('暂停记录', '暂停记录', false), '暂停记录');
});

test('kws enter adds tags, rejects duplicates across categories and previews pronunciation', async () => {
    const c = kwsClient({
        'GET /v1/kws/keywords': () => [200, kwsDoc],
        'POST /v1/kws/keywords/preview': body => [200, { items: body.phrases.map(phrase => ({ phrase, tokens: 'sh í w ǔ ch uáng', error: null })) }],
    });
    await c.scope.loadKwsKeywords();
    c.scope.kwsInputKey(key('Enter', '12床、15床 下一床'), 'bed');
    assert.deepEqual(JSON.parse(c.get("JSON.stringify(kwsCategory('bed').keywords.map(k => [k.phrase, k.value]))")), [['十二床', '12床'], ['十五床', '15床']]);
    assert.equal(c.get("kwsUi.hints.bed"), '「十二床」已在床号里；「下一床」已在查房指令里');
    assert.equal(c.scope.kwsChangeCount(), 1);
    await settle();
    assert.deepEqual(c.calls[1].body, { phrases: ['十五床'] });
    assert.equal(c.get("kwsCategory('bed').keywords[1].tokens"), 'sh í w ǔ ch uáng');
    assert.ok(c.elements.get('kwsCategories').innerHTML.includes('十五床'));
    assert.ok(c.elements.get('kwsSaveText').innerHTML.includes('1 处改动未保存'));
});

test('kws backspace arms then deletes the last tag and undo restores it', async () => {
    const c = kwsClient({ 'GET /v1/kws/keywords': () => [200, kwsDoc] });
    await c.scope.loadKwsKeywords();
    c.scope.kwsInputKey(key('Backspace'), 'command');
    assert.equal(c.get("kwsCategory('command').keywords.length"), 1);
    c.scope.kwsInputKey(key('Backspace'), 'command');
    assert.equal(c.get("kwsCategory('command').keywords.length"), 0);
    assert.equal(c.scope.kwsChangeCount(), 1);
    c.scope.kwsUndoLast();
    assert.equal(c.scope.kwsChangeCount(), 0);
});

test('kws save sends the read version and keeps the draft on conflict', async () => {
    let attempt = 0;
    const c = kwsClient({
        'GET /v1/kws/keywords': () => [200, kwsDoc],
        'PUT /v1/kws/keywords': body => (++attempt === 1
            ? [409, { detail: '关键词已被其他人修改，请刷新后重试' }]
            : [200, { ...kwsDoc, version: body.version + 1, categories: body.categories }]),
    });
    await c.scope.loadKwsKeywords();
    c.scope.kwsSetSlot('command', true);
    c.scope.kwsEditField(c.get("kwsCategory('bed').keywords[0].id"), 'boost', '1.5');
    await c.scope.saveKwsKeywords();
    assert.equal(c.calls[1].body.version, 3);
    assert.deepEqual(c.calls[1].body.categories[0].keywords[0], { phrase: '十二床', value: '12床', boost: '1.5', threshold: '' });
    assert.equal(c.calls[1].body.categories[1].keywords[0].threshold, '0.3');
    assert.ok(c.get('kwsUi.saveError').includes('关键词已被其他人修改'));
    assert.equal(c.scope.kwsChangeCount(), 2);
    await c.scope.saveKwsKeywords();
    assert.equal(c.get('kwsMeta.version'), 4);
    assert.equal(c.scope.kwsChangeCount(), 0);
});

test('kws blocks saving unrecognized phrases and reports a disabled server', async () => {
    const c = kwsClient({
        'GET /v1/kws/keywords': () => [200, kwsDoc],
        'POST /v1/kws/keywords/preview': body => [200, { items: body.phrases.map(phrase => ({ phrase, tokens: '', error: '发音 B 不在模型词表中' })) }],
    });
    await c.scope.loadKwsKeywords();
    c.scope.kwsAddPhrases('command', 'B超');
    await settle();
    await c.scope.saveKwsKeywords();
    assert.ok(c.get('kwsUi.saveError').includes('B超'));
    assert.equal(c.calls.filter(call => call.method === 'PUT').length, 0);

    const disabled = kwsClient({ 'GET /v1/kws/keywords': () => [200, { enabled: false }] });
    await disabled.scope.loadKwsKeywords();
    assert.ok(disabled.elements.get('kwsCategories').innerHTML.includes('KWS_ENABLED=0'));
});
