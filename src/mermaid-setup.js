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
                    theme: 'dark',
                    securityLevel: 'loose',
                    themeVariables: {
                        primaryColor: '#2c3e50',
                        primaryTextColor: '#ecf0f1',
                        primaryBorderColor: '#3498db',
                        lineColor: '#bdc3c7',
                        secondaryColor: '#34495e',
                        tertiaryColor: '#1a252f',
                        fontFamily: 'Segoe UI, Roboto, Helvetica, Arial, sans-serif',
                        fontSize: '14px'
                    },
                    flowchart: {
                        useMaxWidth: true,
                        htmlLabels: true,
                        curve: 'basis'
                    },
                    sequence: {
                        useMaxWidth: true,
                        diagramMarginX: 20,
                        diagramMarginY: 20
                    },
                    gantt: {
                        useMaxWidth: true
                    },
                    mindmap: {
                        useMaxWidth: true
                    },
                    timeline: {
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
                // mermaid.init() works in both v9 and v10
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
