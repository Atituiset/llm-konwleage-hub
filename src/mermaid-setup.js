// Mermaid diagram support for mdbook SPA navigation
// mdbook uses AJAX navigation (History API), so we need MutationObserver
// to detect new page content and render mermaid diagrams dynamically.

(function() {
    'use strict';

    var mermaidReady = false;

    function loadMermaidCDN(callback) {
        if (window.mermaidLoaded) {
            if (mermaidReady) callback();
            else setTimeout(function() { loadMermaidCDN(callback); }, 100);
            return;
        }
        window.mermaidLoaded = true;

        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
        script.async = true;
        script.onload = function() {
            if (typeof mermaid !== 'undefined') {
                mermaid.initialize({
                    startOnLoad: false,
                    theme: 'base',
                    securityLevel: 'loose',
                    themeVariables: {
                        // 主节点：亮蓝底 + 白字，确保在 navy 深色背景上醒目
                        primaryColor: '#2980b9',
                        primaryTextColor: '#ffffff',
                        primaryBorderColor: '#5dade2',

                        // 次要节点
                        secondaryColor: '#27ae60',
                        secondaryTextColor: '#ffffff',
                        secondaryBorderColor: '#58d68d',

                        // 第三级节点
                        tertiaryColor: '#8e44ad',
                        tertiaryTextColor: '#ffffff',
                        tertiaryBorderColor: '#af7ac5',

                        // 线条：亮白色，确保清晰可见
                        lineColor: '#ffffff',

                        // 文本和标签
                        textColor: '#ffffff',
                        fontFamily: 'Segoe UI, Roboto, Helvetica, Arial, sans-serif',
                        fontSize: '18px',

                        // 背景
                        background: '#1a2332',
                        mainBkg: '#2c3e50',
                        secondBkg: '#34495e',
                        tertiaryBkg: '#1a252f',

                        // 特殊元素
                        primaryBorderColor: '#5dade2',
                        edgeLabelBackground: '#1a2332',
                        nodeTextColor: '#ffffff',
                        clusterBkg: '#243447',
                        clusterBorder: '#3498db',
                        titleColor: '#ffffff',
                        edgeLabelText: '#ffffff',
                        activeTaskBkgColor: '#2980b9',
                        activeTaskBorderColor: '#5dade2',
                        gridColor: '#34495e',
                        section0: '#2980b9',
                        section1: '#27ae60',
                        section2: '#8e44ad',
                        section3: '#c0392b',
                        task0: '#2980b9',
                        task1: '#27ae60',
                        task2: '#8e44ad',
                        task3: '#c0392b',
                        todayLineColor: '#e74c3c',
                        git0: '#2980b9',
                        git1: '#27ae60',
                        git2: '#8e44ad',
                        git3: '#c0392b',
                        gitBranchLabel0: '#ffffff',
                        gitBranchLabel1: '#ffffff',
                        gitBranchLabel2: '#ffffff',
                        gitBranchLabel3: '#ffffff',
                        commitLabelColor: '#ffffff',
                        commitLabelBackground: '#2c3e50',
                        tagLabelColor: '#ffffff',
                        tagLabelBackground: '#8e44ad',
                        gitInv0: '#34495e',
                        gitInv1: '#2c3e50',
                        gitInv2: '#1a252f',
                        gitInv3: '#243447'
                    },
                    flowchart: {
                        useMaxWidth: true,
                        htmlLabels: true,
                        curve: 'basis',
                        padding: 15
                    },
                    sequence: {
                        useMaxWidth: true,
                        diagramMarginX: 30,
                        diagramMarginY: 30,
                        actorFontSize: 18,
                        noteFontSize: 16,
                        messageFontSize: 16
                    },
                    gantt: {
                        useMaxWidth: true
                    },
                    mindmap: {
                        useMaxWidth: true
                    },
                    timeline: {
                        useMaxWidth: true
                    },
                    er: {
                        useMaxWidth: true
                    },
                    journey: {
                        useMaxWidth: true
                    },
                    gitgraph: {
                        useMaxWidth: true
                    }
                });
                mermaidReady = true;
                callback();
            }
        };
        script.onerror = function() {
            console.error('[Mermaid] Failed to load CDN script');
        };
        document.head.appendChild(script);
    }

    function renderMermaidDiagrams() {
        if (!mermaidReady || typeof mermaid === 'undefined') {
            return;
        }

        var blocks = document.querySelectorAll('pre code.language-mermaid');
        if (!blocks.length) return;

        var hasNew = false;
        blocks.forEach(function(codeBlock) {
            var pre = codeBlock.parentElement;
            if (!pre) return;
            if (pre.getAttribute('data-mermaid-rendered') === 'true') return;

            var graphDefinition = codeBlock.textContent || '';
            if (!graphDefinition.trim()) return;

            var mermaidDiv = document.createElement('div');
            mermaidDiv.className = 'mermaid';
            mermaidDiv.textContent = graphDefinition;

            pre.parentNode.insertBefore(mermaidDiv, pre);
            pre.style.display = 'none';
            pre.setAttribute('data-mermaid-rendered', 'true');
            hasNew = true;
        });

        if (hasNew) {
            try {
                mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            } catch (err) {
                console.error('[Mermaid] init() error:', err);
            }
        }
    }

    function resetRenderedState() {
        document.querySelectorAll('pre[data-mermaid-rendered="true"]').forEach(function(el) {
            el.removeAttribute('data-mermaid-rendered');
            el.style.display = '';
        });
        document.querySelectorAll('div.mermaid').forEach(function(el) {
            el.remove();
        });
    }

    function setupMutationObserver() {
        var contentArea = document.getElementById('content');
        if (!contentArea) contentArea = document.querySelector('main');
        if (!contentArea) contentArea = document.body;

        var observer = new MutationObserver(function(mutations) {
            var shouldRender = false;
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldRender = true;
                }
            });

            if (shouldRender) {
                setTimeout(function() {
                    resetRenderedState();
                    renderMermaidDiagrams();
                }, 50);
            }
        });

        observer.observe(contentArea, { childList: true, subtree: true });
    }

    function hookMdbookPageChange() {
        document.addEventListener('mdbook-page-changed', function() {
            resetRenderedState();
            renderMermaidDiagrams();
        });

        window.addEventListener('popstate', function() {
            setTimeout(function() {
                resetRenderedState();
                renderMermaidDiagrams();
            }, 100);
        });
    }

    function init() {
        loadMermaidCDN(function() {
            setTimeout(renderMermaidDiagrams, 100);
            setupMutationObserver();
            hookMdbookPageChange();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
