/* 模型对决评测平台 · 公共脚本（迭代八：统一 fetch/转义/toast/三态/顶栏/轮询）
   所有页面在自身 <script> 前引入：<script src="/static/common.js"></script>
   CSP：外链脚本经 script-src 'self' 放行；本文件不触碰任何外部数据渲染。 */
(function(){
  'use strict';

  var TOKEN_KEY = 'duel_token';
  var NAV_LINKS = [
    {key: 'tasks',    href: '/tasks.html',      label: '任务调度'},
    {key: 'lb',       href: '/leaderboard.html',label: '排行榜'},
    {key: 'perturb',  href: '/perturb.html',    label: '扰动评测'},
    {key: 'dash',     href: '/dashboard.html',  label: 'KPI 看板'},
    {key: 'badcase',  href: '/badcases.html',   label: 'Bad Case'},
    {key: 'gen',      href: '/gen_review.html', label: '出题审核'},
  ];

  function getToken(){
    try { return localStorage.getItem(TOKEN_KEY) || ''; } catch(e){ return ''; }
  }

  /* 统一 fetch：携带令牌；401 抛错并广播 auth-401（页面可监听提示） */
  async function apiFetch(url, opts){
    const headers = Object.assign({}, opts && opts.headers);
    const token = getToken();
    if(token) headers['Authorization'] = 'Bearer ' + token;
    const r = await fetch(url, Object.assign({}, opts, {headers}));
    if(r.status === 401){
      try { document.dispatchEvent(new CustomEvent('auth-401')); } catch(e){}
      throw new Error('未授权：请在页面顶部设置访问令牌');
    }
    return r;
  }

  function esc(s){
    return String(s == null ? '' : s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }

  /* 全局消息条（替代 alert） */
  var _toastHost = null;
  function toast(msg, kind){
    kind = kind || 'err';
    if(!_toastHost){
      _toastHost = document.createElement('div');
      _toastHost.className = 'toast';
      document.body.appendChild(_toastHost);
    }
    var t = document.createElement('div');
    t.className = 'toast-item ' + kind;
    t.textContent = msg;
    _toastHost.appendChild(t);
    setTimeout(function(){ t.remove(); }, 4000);
  }

  /* 统一顶栏：同步注入 body 最前（先于任何缺参/错误分支，D10） */
  function renderTopNav(active){
    if(document.querySelector('.topnav')) return;
    var nav = document.createElement('nav');
    nav.className = 'topnav';
    var brand = document.createElement('a');
    brand.className = 'brand';
    brand.href = '/';
    brand.textContent = '⚔️ 模型对决评测平台';
    nav.appendChild(brand);
    var links = document.createElement('div');
    links.className = 'topnav-links';
    NAV_LINKS.forEach(function(item){
      var a = document.createElement('a');
      a.href = item.href;
      a.textContent = item.label;
      if(item.key === active) a.className = 'active';
      links.appendChild(a);
    });
    nav.appendChild(links);
    document.body.insertBefore(nav, document.body.firstChild);
  }

  /* 轮询封装：内部 try/catch；连续失败 5 次 toast 提示（不中断轮询） */
  function poll(fn, ms){
    var fails = 0;
    var tick = function(){
      try {
        Promise.resolve(fn()).then(function(){ fails = 0; }, function(){
          fails += 1;
          if(fails === 5) toast('数据刷新连续失败，请检查服务状态', 'warn');
        });
      } catch(e){
        fails += 1;
        if(fails === 5) toast('数据刷新连续失败，请检查服务状态', 'warn');
      }
    };
    tick();
    return setInterval(tick, ms || 3000);
  }

  function setLoading(el, text){
    el.innerHTML = '<span class="spinner"></span> ' + (text || '加载中...');
  }

  function renderEmpty(el, text){
    el.innerHTML = '<div class="empty">' + esc(text || '暂无数据') + '</div>';
  }

  /* 错误态渲染：可选重试回调（按钮走 addEventListener，符合 CSP） */
  function renderError(el, msg, retryFn){
    var box = document.createElement('div');
    box.className = 'error-state';
    var span = document.createElement('span');
    span.textContent = msg || '加载失败';
    box.appendChild(span);
    if(typeof retryFn === 'function'){
      var btn = document.createElement('button');
      btn.className = 'btn btn-secondary';
      btn.textContent = '重试';
      btn.style.padding = '6px 16px';
      btn.addEventListener('click', retryFn);
      box.appendChild(btn);
    }
    el.innerHTML = '';
    el.appendChild(box);
  }

  window.CD = {
    getToken: getToken,
    apiFetch: apiFetch,
    esc: esc,
    toast: toast,
    renderTopNav: renderTopNav,
    poll: poll,
    setLoading: setLoading,
    renderEmpty: renderEmpty,
    renderError: renderError,
  };
})();
