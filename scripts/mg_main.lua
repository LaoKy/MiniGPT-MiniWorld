-- PHAN 0: LOG

local LOG_LEVEL = 2  -- 1=DEBUG 2=INFO 3=WARN 4=ERROR
local function log(lv, tag, msg)
    if lv < LOG_LEVEL then return end
    local p = ({[1]='[DBG]',[2]='[INF]',[3]='[WRN]',[4]='[ERR]'})[lv] or '[???]'
    print(string.format('%s[%s] %s', p, tag, tostring(msg)))
end
local function logD(t,m) log(1,t,m) end
local function logI(t,m) log(2,t,m) end
local function logW(t,m) log(3,t,m) end
local function logE(t,m) log(4,t,m) end

local function getTime()
    local ok,t = pcall(function() return os.clock() end)
    return ok and t or 0
end
local function elapsed(t0) return string.format('%.2fs', getTime()-t0) end

logI('SYS', '=== mg_main v5 loading ===')

-- PHAN 1: TOKENIZER

local function makeTokenizer()
    if not MGPT_TOK then logE('TOK','MGPT_TOK nil!'); return nil end
    local tok = {}
    tok.vocab      = MGPT_TOK.vocab
    tok.inv_vocab  = {}
    tok.PAD = MGPT_TOK.PAD or 0
    tok.UNK = MGPT_TOK.UNK or 1
    tok.BOS = MGPT_TOK.BOS or 2
    tok.EOS = MGPT_TOK.EOS or 3
    tok.SEP = MGPT_TOK.SEP or 4
    tok.merges_hash = MGPT_TOK.merges or {}

    local n = 0
    for token, id in pairs(tok.vocab) do
        tok.inv_vocab[id] = token
        n = n + 1
    end
    logI('TOK', string.format('vocab=%d merges=%d', n,
        (function() local c=0; for _ in pairs(tok.merges_hash) do c=c+1 end; return c end)()))

    local function bpe_word(chars)
        local changed = true
        while changed and #chars > 1 do
            changed = false
            local best_rank, best_pos = 999999, -1
            for i = 1, #chars-1 do
                local rank = tok.merges_hash[chars[i]..'#'..chars[i+1]]
                if rank and rank < best_rank then
                    best_rank, best_pos = rank, i
                end
            end
            if best_pos > 0 then
                local nc = {}
                for i = 1, best_pos-1 do nc[#nc+1] = chars[i] end
                nc[#nc+1] = chars[best_pos]..chars[best_pos+1]
                for i = best_pos+2, #chars do nc[#nc+1] = chars[i] end
                chars   = nc
                changed = true
            end
        end
        return chars
    end

    local function utf8_chars(s)
        local chars = {}
        local i = 1
        local len = #s
        while i <= len do
            local b = string.byte(s, i)
            local n = 1
            if     b >= 240 then n = 4
            elseif b >= 224 then n = 3
            elseif b >= 192 then n = 2
            end
            chars[#chars+1] = string.sub(s, i, i + n - 1)
            i = i + n
        end
        return chars
    end

    local function safe_lower(s)
        local result = {}
        local i = 1
        local len = #s
        while i <= len do
            local b = string.byte(s, i)
            if b >= 65 and b <= 90 then
                -- Chi lowercase ASCII A-Z
                result[#result+1] = string.char(b + 32)
                i = i + 1
            elseif b >= 192 then
                -- UTF-8 multibyte: giu nguyen
                local n = 2
                if b >= 240 then n = 4
                elseif b >= 224 then n = 3
                end
                result[#result+1] = string.sub(s, i, i + n - 1)
                i = i + n
            else
                result[#result+1] = string.char(b)
                i = i + 1
            end
        end
        return table.concat(result)
    end

    local function encode_raw(text)
        text = safe_lower(text)
        local ids = {}
        for word in text:gmatch('%S+') do
            local chars = utf8_chars(word)
            chars[#chars+1] = '</w>'
            for _, token in ipairs(bpe_word(chars)) do
                ids[#ids+1] = tok.vocab[token] or tok.UNK
            end
        end
        return ids
    end

    function tok:encodeQ(text)
        local ids = encode_raw(text)
        local result = {self.BOS}
        for _, id in ipairs(ids) do result[#result+1] = id end
        result[#result+1] = self.SEP
        return result
    end

    function tok:decodeAnswer(ids)
        local sep_pos = nil
        for i, id in ipairs(ids) do
            if id == self.SEP then sep_pos = i; break end
        end
        local start = sep_pos and (sep_pos + 1) or 1
        local tokens = {}
        for i = start, #ids do
            local id = ids[i]
            if id ~= self.BOS and id ~= self.EOS
               and id ~= self.PAD and id ~= self.SEP then
                local t = self.inv_vocab[id]
                if t then tokens[#tokens+1] = t end
            end
        end
        return table.concat(tokens,''):gsub('</w>',' '):match('^%s*(.-)%s*$')
    end

    logI('TOK', 'OK')
    return tok
end

-- PHAN 2: LOAD WEIGHT

local BANKS        = nil
local weight_cache = {}
local mat_cache    = {}
local weight_missing = {}

local function initBanks()
    if BANKS then return end
    BANKS = {}
    for i = 0, 29 do
        local found = false
        for t = 0, 9 do
            local b = _G['MGPT_W'..i..'_'..t]
            if b then
                BANKS[#BANKS+1] = b
                logI('LOADER', 'MGPT_W'..i..'_'..t..' OK')
                found = true
            end
        end
        if not found then
            local b = _G['MGPT_W'..i]
            if b then BANKS[#BANKS+1] = b; logI('LOADER','MGPT_W'..i..' OK') end
        end
    end
    logI('LOADER', #BANKS..' banks')
end

local function getFlat(name)
    if weight_cache[name] then return weight_cache[name] end
    if weight_missing[name] then return nil end
    initBanks()

    -- Dequantize weight chunk
    -- Per-row:    raw.rows, raw.cols, raw.s[]  -> v * s[row] / 127
    -- Per-tensor: raw.scale                   -> v * scale / 127
    local function readChunk(bank, key)
        local raw = bank[key]
        if not raw then return nil end
        local result = {}

        if raw.rows and raw.s then
            -- Per-row: moi hang co scale rieng -> chinh xac hon ~10x
            local R = raw.rows
            local C = raw.cols
            for r = 1, R do
                local factor = raw.s[r] / 127.0
                local base   = (r-1) * C
                for c = 1, C do
                    result[base + c] = raw.d[base + c] * factor
                end
            end
        else
            -- Per-tensor: 1D vectors (norm weights, rope, ...)
            local factor = raw.scale / 127.0
            for _, v in ipairs(raw.d) do
                result[#result+1] = v * factor
            end
        end

        return result
    end

    local function getBank(chunkName)
        if not MGPT_IDX then return nil end
        local fidx = MGPT_IDX.files  and MGPT_IDX.files[chunkName]
        local tidx = MGPT_IDX.tables and MGPT_IDX.tables[chunkName]
        if fidx == nil then return nil end
        if tidx ~= nil then
            local b = _G['MGPT_W'..fidx..'_'..tidx]
            if b then return b end
        end
        return _G['MGPT_W'..fidx]
    end

    -- Test truc tiep
    local bank = getBank(name)
    if bank then
        local r = readChunk(bank, name)
        if r then weight_cache[name] = r; return r end
    end

    -- Thu multi-part (embed_weight_p0, p1, ...)
    local parts = {}
    local i = 0
    while true do
        local pname = name..'_p'..i
        local pb = getBank(pname)
        if not pb then break end
        local chunk = readChunk(pb, pname)
        if not chunk then break end
        for _, v in ipairs(chunk) do parts[#parts+1] = v end
        i = i + 1
    end
    if #parts > 0 then
        weight_cache[name] = parts; return parts
    end

    -- Fallback
    for _, bk in ipairs(BANKS) do
        if bk[name] then
            local r = readChunk(bk, name)
            if r then weight_cache[name] = r; return r end
        end
    end

    weight_missing[name] = true
    if not name:match('bias$') then
        logE('LOADER', 'NOT FOUND: '..name)
    end
    return nil
end

local function getMat(name, rows, cols)
    local ck = name..'#'..rows..'x'..cols
    if mat_cache[ck] then return mat_cache[ck] end
    local flat = getFlat(name)
    if not flat then return nil end
    local mat = {}
    for r = 1, rows do
        local row = {}
        local base = (r-1)*cols
        for c = 1, cols do row[c] = flat[base+c] or 0 end
        mat[r] = row
    end
    mat_cache[ck] = mat
    return mat
end

local function preloadMatrices()
    if not cfg then return end
    local D, F, L = cfg.embed_dim, cfg.ffn_dim, cfg.n_layers
    local KVH = cfg.n_kv_heads
    local HD  = math.floor(D / cfg.n_heads)
    for li = 0, L-1 do
        getMat('blocks_'..li..'_attn_wq_weight', D,      D)
        getMat('blocks_'..li..'_attn_wk_weight', KVH*HD, D)
        getMat('blocks_'..li..'_attn_wv_weight', KVH*HD, D)
        getMat('blocks_'..li..'_attn_wo_weight', D,      D)
        getMat('blocks_'..li..'_ffn_w1_weight',  F, D)
        getMat('blocks_'..li..'_ffn_w2_weight',  D, F)
        getMat('blocks_'..li..'_ffn_w3_weight',  F, D)
        getFlat('blocks_'..li..'_norm1_weight')
        getFlat('blocks_'..li..'_norm2_weight')
    end
    getFlat('norm_weight')
    getEmb()
    logI('LOADER', 'mat_cache preload OK: '..tostring((function() local c=0 for _ in pairs(mat_cache) do c=c+1 end return c end)())..' mats')
end

-- PHAN 3: CAC CONG THUC TOAN HOC

local function matvec(A, x, rows, cols)
    local y = {}
    for i = 1, rows do
        local s, Ai = 0, A[i]
        for j = 1, cols do s = s + Ai[j]*x[j] end
        y[i] = s
    end
    return y
end

local function softmax(x, n)
    local mx = x[1]
    for i = 2, n do if x[i] > mx then mx = x[i] end end
    local s, r = 0, {}
    for i = 1, n do r[i] = math.exp(x[i]-mx); s = s+r[i] end
    local inv = 1.0/s
    for i = 1, n do r[i] = r[i]*inv end
    return r
end

local function rmsnorm(x, w, n)
    local ss = 0
    for i = 1, n do ss = ss + x[i]*x[i] end
    local inv = 1.0 / math.sqrt(ss/n + 1e-5)
    local r = {}
    for i = 1, n do r[i] = x[i]*inv*w[i] end
    return r
end

local function layernorm(x, w, b, n)
    local mean = 0
    for i = 1, n do mean = mean + x[i] end
    mean = mean / n
    local var = 0
    for i = 1, n do local d = x[i]-mean; var = var+d*d end
    local inv = 1.0 / math.sqrt(var/n + 1e-5)
    local r = {}
    if b and #b >= n then
        for i = 1, n do r[i] = (x[i]-mean)*inv*w[i]+b[i] end
    else
        for i = 1, n do r[i] = (x[i]-mean)*inv*w[i] end
    end
    return r
end

local function silu(x) return x / (1.0 + math.exp(-x)) end

local function addv(a, b, n)
    local r = {}
    for i = 1, n do r[i] = a[i]+b[i] end
    return r
end

local function sample_topk(logits, n, temp, k)
    temp = temp or 0.7
    k    = k    or 15
    local list = {}
    for i = 1, n do list[i] = {i, logits[i]} end
    table.sort(list, function(a,b) return a[2] > b[2] end)
    local scaled, idxmap = {}, {}
    local kk = math.min(k, n)
    for i = 1, kk do
        scaled[i] = list[i][2] / temp
        idxmap[i] = list[i][1]
    end
    local probs = softmax(scaled, kk)
    local r, sum = math.random(), 0
    for i = 1, kk do
        sum = sum + probs[i]
        if r <= sum then return idxmap[i] end
    end
    return idxmap[1]
end

-- PHAN 4: INFERENCE

local function sample_greedy(logits, n)
    local best, bi = logits[1], 1
    for i = 2, n do
        if logits[i] > best then best, bi = logits[i], i end
    end
    return bi
end

local cfg      = nil
local emb_flat = nil

local function initCfg()
    if cfg then return true end
    if not MGPT_IDX then logE('CFG','MGPT_IDX nil!'); return false end
    cfg = MGPT_IDX.cfg
    logI('CFG', string.format('vocab=%d embed=%d ffn=%d layers=%d heads=%d kv=%d ctx=%d',
        cfg.vocab_size, cfg.embed_dim, cfg.ffn_dim,
        cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ctx_len))
    return true
end

local function getEmb()
    if not emb_flat then
        emb_flat = getFlat('embed_weight')
        if emb_flat then logI('EMB','OK '..#emb_flat..' vals')
        else logE('EMB','FAILED') end
    end
    return emb_flat
end

local function embedToken(tid)
    local D   = cfg.embed_dim
    local emb = getEmb()
    local vec = {}
    local off = tid * D
    for i = 1, D do vec[i] = emb[off+i] or 0 end
    return vec
end

local rope_cache = {}
local function getRope(li)
    local key = 'L'..li
    if rope_cache[key] then return rope_cache[key] end
    local HD       = math.floor(cfg.embed_dim / cfg.n_heads)
    local cos_flat = getFlat('blocks_'..li..'_attn_rope_cos')
    local sin_flat = getFlat('blocks_'..li..'_attn_rope_sin')
    rope_cache[key] = {cos=cos_flat, sin=sin_flat, HD=HD}
    return rope_cache[key]
end

local function applyRope(x, pos, rope)
    local HD   = rope.HD
    local half = math.floor(HD/2)
    local cos  = rope.cos
    local sin  = rope.sin
    if not cos or not sin then return x end
    local off    = (pos-1) * HD
    local result = {}
    for i = 1, half do
        local c  = cos[off+i] or 1
        local s  = sin[off+i] or 0
        local xi  = x[i]      or 0
        local xi2 = x[i+half] or 0
        result[i]      = xi*c  - xi2*s
        result[i+half] = xi*s  + xi2*c
    end
    return result
end

local function ffn(x, li)
    local D = cfg.embed_dim
    local F = cfg.ffn_dim
    local p = 'blocks_'..li..'_ffn_'
    local w1 = getMat(p..'w1_weight', F, D)
    local w2 = getMat(p..'w2_weight', D, F)
    local w3 = getMat(p..'w3_weight', F, D)
    if not w1 or not w2 or not w3 then return x end
    local gate = matvec(w1, x, F, D)
    local up   = matvec(w3, x, F, D)
    local mid  = {}
    for i = 1, F do mid[i] = silu(gate[i]) * up[i] end
    return matvec(w2, mid, D, F)
end

local function attn(xs, li)
    local D   = cfg.embed_dim
    local H   = cfg.n_heads
    local KVH = cfg.n_kv_heads
    local HD  = math.floor(D/H)
    local T   = #xs
    local p   = 'blocks_'..li..'_attn_'
    local wq  = getMat(p..'wq_weight', D,      D)
    local wk  = getMat(p..'wk_weight', KVH*HD, D)
    local wv  = getMat(p..'wv_weight', KVH*HD, D)
    local wo  = getMat(p..'wo_weight', D,      D)
    if not wq or not wk or not wv or not wo then return xs end

    local rope = getRope(li)
    local sc   = 1.0 / math.sqrt(HD)
    local Qs, Ks, Vs = {}, {}, {}

    for t = 1, T do
        local qf = matvec(wq, xs[t], D,      D)
        local kf = matvec(wk, xs[t], KVH*HD, D)
        local vf = matvec(wv, xs[t], KVH*HD, D)
        Qs[t] = {}; Ks[t] = {}
        for h = 1, H do
            local hs = (h-1)*HD+1
            local qh = {}
            for d = 1, HD do qh[d] = qf[hs+d-1] end
            local qr = applyRope(qh, t, rope)
            for d = 1, HD do Qs[t][hs+d-1] = qr[d] end
        end
        for h = 1, KVH do
            local hs = (h-1)*HD+1
            local kh = {}
            for d = 1, HD do kh[d] = kf[hs+d-1] end
            local kr = applyRope(kh, t, rope)
            for d = 1, HD do Ks[t][hs+d-1] = kr[d] end
        end
        Vs[t] = vf
    end

    local out = {}
    for t = 1, T do out[t] = {}; for d = 1, D do out[t][d] = 0 end end

    for h = 1, H do
        local kvh    = math.floor((h-1)*KVH/H) + 1
        local hstart = (h-1)*HD + 1
        local kstart = (kvh-1)*HD + 1
        for t = 1, T do
            local q = {}
            for d = 1, HD do q[d] = Qs[t][hstart+d-1] end
            local scores = {}
            for s = 1, t do
                local dot = 0
                for d = 1, HD do dot = dot + q[d]*Ks[s][kstart+d-1] end
                scores[s] = dot * sc
            end
            local a = softmax(scores, t)
            for s = 1, t do
                local w = a[s]
                for d = 1, HD do
                    out[t][hstart+d-1] = out[t][hstart+d-1] + w*Vs[s][kstart+d-1]
                end
            end
        end
    end

    local res = {}
    for t = 1, T do res[t] = matvec(wo, out[t], D, D) end
    return res
end

local function block(xs, li)
    local D = cfg.embed_dim
    local T = #xs
    local p = 'blocks_'..li..'_'
    local n1w = getFlat(p..'norm1_weight')
    local n1b = getFlat(p..'norm1_bias')
    local n2w = getFlat(p..'norm2_weight')
    local n2b = getFlat(p..'norm2_bias')
    if not n1w or not n2w then
        logE('BLK','layer '..li..' norm missing'); return xs
    end
    local normed1 = {}
    for t = 1, T do
        normed1[t] = n1b and layernorm(xs[t],n1w,n1b,D) or rmsnorm(xs[t],n1w,D)
    end
    local ao = attn(normed1, li)
    local h1 = {}
    for t = 1, T do h1[t] = addv(xs[t], ao[t], D) end
    local res = {}
    for t = 1, T do
        local n2 = n2b and layernorm(h1[t],n2w,n2b,D) or rmsnorm(h1[t],n2w,D)
        res[t] = addv(h1[t], ffn(n2, li), D)
    end
    return res
end

-- PHAN 4b: KV-CACHE
local kv_cache = {}

local function kv_reset() kv_cache = {} end

local function attn_step(x, li, pos)
    local D   = cfg.embed_dim
    local H   = cfg.n_heads
    local KVH = cfg.n_kv_heads
    local HD  = math.floor(D/H)
    local p   = 'blocks_'..li..'_attn_'
    local wq  = getMat(p..'wq_weight', D,      D)
    local wk  = getMat(p..'wk_weight', KVH*HD, D)
    local wv  = getMat(p..'wv_weight', KVH*HD, D)
    local wo  = getMat(p..'wo_weight', D,      D)
    if not wq or not wk or not wv or not wo then return nil end

    local rope = getRope(li)
    local sc   = 1.0 / math.sqrt(HD)

    local qf = matvec(wq, x, D,      D)
    local kf = matvec(wk, x, KVH*HD, D)
    local vf = matvec(wv, x, KVH*HD, D)

    local q = {}
    for h = 1, H do
        local hs = (h-1)*HD+1
        local qh = {}
        for d = 1, HD do qh[d] = qf[hs+d-1] end
        local qr = applyRope(qh, pos, rope)
        for d = 1, HD do q[hs+d-1] = qr[d] end
    end
    local k = {}
    for h = 1, KVH do
        local hs = (h-1)*HD+1
        local kh = {}
        for d = 1, HD do kh[d] = kf[hs+d-1] end
        local kr = applyRope(kh, pos, rope)
        for d = 1, HD do k[hs+d-1] = kr[d] end
    end

    local c = kv_cache[li]
    if not c then c = {K={}, V={}, len=0}; kv_cache[li] = c end
    c.len = c.len + 1
    c.K[c.len] = k
    c.V[c.len] = vf

    local out = {}
    for d = 1, D do out[d] = 0 end
    for h = 1, H do
        local kvh    = math.floor((h-1)*KVH/H) + 1
        local hstart = (h-1)*HD + 1
        local kstart = (kvh-1)*HD + 1
        local qh = {}
        for d = 1, HD do qh[d] = q[hstart+d-1] end
        local scores = {}
        for s = 1, c.len do
            local dot = 0
            local Ks = c.K[s]
            for d = 1, HD do dot = dot + qh[d]*Ks[kstart+d-1] end
            scores[s] = dot * sc
        end
        local a = softmax(scores, c.len)
        for s = 1, c.len do
            local w = a[s]
            local Vs = c.V[s]
            for d = 1, HD do
                out[hstart+d-1] = out[hstart+d-1] + w*Vs[kstart+d-1]
            end
        end
    end
    return matvec(wo, out, D, D)
end

local function block_step(x, li, pos)
    local D = cfg.embed_dim
    local p = 'blocks_'..li..'_'
    local n1w = getFlat(p..'norm1_weight')
    local n1b = getFlat(p..'norm1_bias')
    local n2w = getFlat(p..'norm2_weight')
    local n2b = getFlat(p..'norm2_bias')
    if not n1w or not n2w then return nil end
    local n1 = n1b and layernorm(x,n1w,n1b,D) or rmsnorm(x,n1w,D)
    local ao = attn_step(n1, li, pos)
    if not ao then return nil end
    local h1 = addv(x, ao, D)
    local n2 = n2b and layernorm(h1,n2w,n2b,D) or rmsnorm(h1,n2w,D)
    return addv(h1, ffn(n2, li), D)
end

-- PHAN 5: GENERATE

local g_history    = {}
local MAX_HISTORY  = 1

local function generate_async(input_ids, max_new, tok_ref)
    max_new = max_new or 30
    local D   = cfg.embed_dim
    local V   = cfg.vocab_size
    local L   = cfg.n_layers
    local CTX = cfg.ctx_len
    local t0  = getTime()

    local nw = getFlat('norm_weight')
    local nb = getFlat('norm_bias')
    if not nw then
        nw = {}; for i = 1, D do nw[i] = 1.0 end
        logW('GEN','dung identity norm')
    end

    local emb = getEmb()
    if not emb then logE('GEN','emb nil'); return 'loi he thong' end

    local tokens = {}
    for _, entry in ipairs(g_history) do
        for _, id in ipairs(entry) do tokens[#tokens+1] = id end
    end
    for _, id in ipairs(input_ids) do tokens[#tokens+1] = id end
    if #tokens > CTX then
        local trim = {}
        for i = #tokens-CTX+1, #tokens do trim[#trim+1] = tokens[i] end
        tokens = trim
    end

    local input_len = #tokens

    kv_reset()
    local last_h = nil
    local function prefill()
        for t = 1, #tokens do
            local h = embedToken(tokens[t])
            for l = 0, L-1 do h = block_step(h, l, t) end
            if t == #tokens then last_h = h end
            if t % 8 == 0 then coroutine.yield() end
        end
    end
    prefill()

    for step = 1, max_new do
        local last = nb and layernorm(last_h,nw,nb,D) or rmsnorm(last_h,nw,D)

        -- Tinh logits (tied embeddings)
        local logits = {}
        for v = 0, V-1 do
            local dot = 0
            local off = v * D
            for d = 1, D do dot = dot + last[d] * (emb[off+d] or 0) end
            logits[v+1] = dot
        end

        local win = math.max(1, #tokens-6+1)
        for i = win, #tokens do
            local tid = tokens[i]
            if tid ~= tok_ref.BOS and tid ~= tok_ref.SEP then
                logits[tid+1] = logits[tid+1] - 1.2
            end
        end

        local nid
        if g_fast then
            nid = sample_greedy(logits, V) - 1
        else
            nid = sample_topk(logits, V, g_temperature, 15) - 1
        end
        tokens[#tokens+1] = nid

        if nid == tok_ref.EOS or nid == tok_ref.PAD then break end

        if step >= 3 then
            local last_tok = tokens[#tokens]
            local rep = 0
            for i = #tokens-3, #tokens do
                if tokens[i] == last_tok then rep = rep+1 end
            end
            if rep >= 4 then break end
        end

        if #tokens > CTX then
            local trim = {}
            for i = #tokens-CTX+1, #tokens do trim[#trim+1] = tokens[i] end
            tokens = trim
            kv_reset()
            prefill()
        else
            local h = embedToken(nid)
            for l = 0, L-1 do h = block_step(h, l, #tokens) end
            last_h = h
        end
        coroutine.yield()
    end

    logI('GEN', 'done '..elapsed(t0))

    local answer_ids = {}
    for i = input_len+1, #tokens do answer_ids[#answer_ids+1] = tokens[i] end

    local resp = tok_ref:decodeAnswer(answer_ids)
    if not resp or #resp < 2 then
        resp = 'xin loi minh chua hieu cau hoi nay ban oi'
    end

    local entry = {}
    for _, id in ipairs(input_ids)  do entry[#entry+1] = id end
    for _, id in ipairs(answer_ids) do entry[#entry+1] = id end
    entry[#entry+1] = tok_ref.EOS
    g_history[#g_history+1] = entry
    if #g_history > MAX_HISTORY then table.remove(g_history, 1) end

    return resp
end

-- PHAN 6: CHAT

local g_tok         = nil
local g_ready       = false
g_busy              = false
g_coroutine         = nil
g_temperature       = 0.7
g_fast              = true

local function chat(msg)
    Chat:sendSystemMsg('[MiniGPT] '..tostring(msg))
end

local function initAI()
    if g_busy then chat('Dang xu ly...'); return end
    g_busy = true
    local t0 = getTime()
    chat('Dang khoi dong AI...')

    local n = 0
    for i = 0, 29 do
        if _G['MGPT_W'..i] or _G['MGPT_W'..i..'_0'] then n = n+1 end
    end
    if n == 0        then chat('Loi: khong co weight!'); g_busy=false; return end
    if not MGPT_TOK  then chat('Loi: thieu mg_tok');    g_busy=false; return end
    if not MGPT_IDX  then chat('Loi: thieu mg_idx');    g_busy=false; return end
    if not initCfg() then chat('Loi: config');          g_busy=false; return end

    local ok, err = pcall(function() g_tok = makeTokenizer() end)
    if not ok or not g_tok then
        chat('Loi tokenizer: '..tostring(err)); g_busy=false; return
    end

    getEmb()
    preloadMatrices()
    g_history = {}
    g_ready   = true
    g_busy    = false
    logI('INIT', 'ready '..elapsed(t0))
    chat(string.format('San sang! vocab=%d embed=%d layers=%d',
        cfg.vocab_size, cfg.embed_dim, cfg.n_layers))
    chat('Dung "!ai [cau hoi]" de chat.')
end

local function onChat(e)
    local msg = e.content or ''
    if msg:sub(1,4) ~= '!ai ' then return end
    local input = msg:sub(5):match('^%s*(.-)%s*$')
    if input == '' then chat('Dung: !ai [cau hoi]'); return end

    if input == 'init'      then initAI(); return end
    if input == 'reset'     then
        g_ready=false; g_tok=nil; emb_flat=nil
        weight_cache={}; mat_cache={}; weight_missing={}; kv_reset(); BANKS=nil; rope_cache={}
        g_coroutine=nil; g_busy=false; g_history={}
        chat('Da reset.'); return
    end
    if input == 'clear'     then g_history={}; chat('Da xoa lich su.'); return end
    if input == 'help'      then
        chat('Lenh: init | reset | clear | help | nhanh | chuan | temp low/mid/high | log debug/info/off')
        chat('nhanh = greedy 1 vong, max 16 token (mac dinh, toc do). chuan = topk cu, cham hon.')
        chat('Debug: !ai debug [cau hoi] — in token IDs va ket qua generate')
        return
    end
    if input == 'nhanh'     then g_fast=true;  chat('Che do: nhanh (greedy, max 16)'); return end
    if input == 'chuan'     then g_fast=false; chat('Che do: chuan (topk-15, max 16)'); return end

    if input:sub(1,6) == 'debug ' then
        if not g_ready then chat('Chua init. Dung "!ai init" truoc.'); return end
        local q = input:sub(7):match('^%s*(.-)%s*$')
        if q == '' then chat('Dung: !ai debug [cau hoi]'); return end

        local ids = g_tok:encodeQ(q)
        local id_strs = {}
        for _, id in ipairs(ids) do id_strs[#id_strs+1] = tostring(id) end
        chat(string.format('[DBG] encodeQ("%s")', q))
        chat('[DBG] IDs: '..table.concat(id_strs, ','))

        local tok_strs = {}
        for _, id in ipairs(ids) do
            local t = g_tok.inv_vocab[id]
            if     id == g_tok.BOS then tok_strs[#tok_strs+1] = '<BOS>'
            elseif id == g_tok.EOS then tok_strs[#tok_strs+1] = '<EOS>'
            elseif id == g_tok.SEP then tok_strs[#tok_strs+1] = '<SEP>'
            elseif id == g_tok.PAD then tok_strs[#tok_strs+1] = '<PAD>'
            elseif id == g_tok.UNK then tok_strs[#tok_strs+1] = '<UNK>'
            elseif t then tok_strs[#tok_strs+1] = '['..t..']'
            else        tok_strs[#tok_strs+1] = '[?'..id..']'
            end
        end
        chat('[DBG] Tokens: '..table.concat(tok_strs, ' '))

        if g_busy then chat('[DBG] Dang ban, khong the debug ngay'); return end
        g_busy = true
        chat('[DBG] Dang generate...')

        local input_ids = ids
        local tok_ref   = g_tok
        g_coroutine = coroutine.create(function()
            local D   = cfg.embed_dim
            local V   = cfg.vocab_size
            local L   = cfg.n_layers
            local CTX = cfg.ctx_len

            local nw = getFlat('norm_weight')
            local nb = getFlat('norm_bias')
            if not nw then
                nw = {}; for i = 1, D do nw[i] = 1.0 end
            end
            local emb = getEmb()
            if not emb then chat('[DBG] FATAL: emb nil'); g_busy=false; return end

            local tokens = {}
            for _, id in ipairs(input_ids) do tokens[#tokens+1] = id end
            local input_len = #tokens

            local generated = {}
            for step = 1, 12 do
                local start = math.max(1, #tokens-CTX+1)
                local ctx = {}
                for i = start, #tokens do ctx[#ctx+1] = tokens[i] end

                local xs = {}
                for _, tid in ipairs(ctx) do xs[#xs+1] = embedToken(tid) end
                coroutine.yield()

                for l = 0, L-1 do xs = block(xs, l); coroutine.yield() end

                local last = nb and layernorm(xs[#xs],nw,nb,D) or rmsnorm(xs[#xs],nw,D)
                local logits = {}
                for v = 0, V-1 do
                    local dot = 0; local off = v*D
                    for d = 1, D do dot = dot + last[d]*(emb[off+d] or 0) end
                    logits[v+1] = dot
                end

                local top5 = {}
                local used = {}
                for k = 1, math.min(5, V) do
                    local bi, bv = 1, logits[1]
                    for v = 2, V do
                        if not used[v] and logits[v] > bv then bi, bv = v, logits[v] end
                    end
                    used[bi] = true
                    local tstr = tok_ref.inv_vocab[bi-1] or ('id'..(bi-1))
                    top5[#top5+1] = string.format('%s(%.2f)', tstr, bv)
                end
                chat(string.format('[DBG] step%d top5: %s', step, table.concat(top5, ' | ')))

                local nid = sample_topk(logits, V, g_temperature, 15) - 1
                tokens[#tokens+1] = nid

                local tstr = tok_ref.inv_vocab[nid] or ('?'..nid)
                generated[#generated+1] = string.format('%d(%s)', nid, tstr)
                coroutine.yield()

                if nid == tok_ref.EOS or nid == tok_ref.PAD then
                    chat('[DBG] -> EOS/PAD, dung')
                    break
                end
            end

            chat('[DBG] Generated IDs: '..table.concat(generated, ' '))

            local answer_ids = {}
            for i = input_len+1, #tokens do answer_ids[#answer_ids+1] = tokens[i] end
            local resp = tok_ref:decodeAnswer(answer_ids)
            chat('[DBG] Ket qua: "'..(resp or '(trong)')..'\"')
            g_busy = false
        end)
        return
    end
    if input == 'log debug' then LOG_LEVEL=1; chat('Log: DEBUG'); return end
    if input == 'log info'  then LOG_LEVEL=2; chat('Log: INFO');  return end
    if input == 'log off'   then LOG_LEVEL=4; chat('Log: OFF');   return end
    if input == 'temp low'  then g_temperature=0.5; chat('Temp: 0.5'); return end
    if input == 'temp mid'  then g_temperature=0.7; chat('Temp: 0.7'); return end
    if input == 'temp high' then g_temperature=0.9; chat('Temp: 0.9'); return end

    if input:sub(1,5) == 'tinh ' then
        local expr = input:sub(6):gsub('[xX]','*')
        local ok, result = pcall(function()
            return (load('return '..expr))()
        end)
        chat(ok and result ~= nil
            and string.format('= %g', result)
            or  'Khong tinh duoc')
        return
    end

    if not g_ready then chat('Chua san sang. Dung "!ai init" truoc.'); return end
    if g_busy      then chat('Dang ban...'); return end

    local ids = g_tok:encodeQ(input)
    logI('CHAT', string.format('"%s" -> %d tokens', input, #ids))

    g_busy = true
    chat('Dang nghi...')

    local input_ids = ids
    local tok_ref   = g_tok
    g_coroutine = coroutine.create(function()
        local ok, result = pcall(generate_async, input_ids, 16, tok_ref)
        if ok then
            chat('AI: '..tostring(result))
        else
            logE('GEN', tostring(result))
            chat('Loi: '..tostring(result))
        end
        g_busy = false
    end)
end

-- PHAN 7: EVENTS + TICK

ScriptSupportEvent:registerEvent([=[Player.NewInputContent]=], onChat)
ScriptSupportEvent:registerEvent([=[Game.Start]=], function()
    Chat:sendSystemMsg('[MiniGPT] Loaded! Dung "!ai init" de bat dau.')
end)

mg_tick = function()
    if g_coroutine == nil then return end
    local status = coroutine.status(g_coroutine)
    if status == 'suspended' then
        local ok, err = coroutine.resume(g_coroutine)
        if not ok then
            logE('TICK', tostring(err))
            Chat:sendSystemMsg('[MiniGPT] Loi: '..tostring(err))
            g_coroutine = nil
            g_busy      = false
        end
    elseif status == 'dead' then
        g_coroutine = nil
    end
end

logI('SYS', '=== mg_main loaded OK (no mg_tick needed) ===')
