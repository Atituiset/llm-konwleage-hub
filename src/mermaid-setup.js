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
                        // 核心背景与文字（所有图表类型通用）
                        background: '#1a2332',
                        textColor: '#ffffff',
                        fontColor: '#ffffff',
                        fontFamily: 'Segoe UI, Roboto, Helvetica, Arial, sans-serif',
                        fontSize: '18px',

                        // 主/次/三级节点颜色
                        primaryColor: '#2980b9',
                        primaryTextColor: '#ffffff',
                        primaryBorderColor: '#5dade2',
                        secondaryColor: '#27ae60',
                        secondaryTextColor: '#ffffff',
                        secondaryBorderColor: '#58d68d',
                        tertiaryColor: '#8e44ad',
                        tertiaryTextColor: '#ffffff',
                        tertiaryBorderColor: '#af7ac5',

                        // 节点通用样式
                        nodeBkg: '#2c3e50',
                        nodeBorder: '#5dade2',
                        nodeTextColor: '#ffffff',
                        mainBkg: '#2c3e50',
                        secondBkg: '#34495e',
                        tertiaryBkg: '#1a252f',

                        // 线条、箭头、连接
                        lineColor: '#ffffff',
                        arrowheadColor: '#ffffff',
                        defaultLinkColor: '#ffffff',

                        // 子图/集群
                        clusterBkg: '#243447',
                        clusterBorder: '#3498db',

                        // 标题与边标签
                        titleColor: '#ffffff',
                        edgeLabelBackground: '#1a2332',
                        edgeLabelText: '#ffffff',
                        labelBackgroundColor: '#1a2332',

                        // Sequence 序列图
                        actorBorder: '#5dade2',
                        actorBkg: '#2980b9',
                        actorTextColor: '#ffffff',
                        actorLineColor: '#ffffff',
                        signalColor: '#ffffff',
                        signalTextColor: '#ffffff',
                        labelBoxBkgColor: '#243447',
                        labelBoxBorderColor: '#3498db',
                        labelTextColor: '#ffffff',
                        loopTextColor: '#ffffff',
                        activationBorderColor: '#5dade2',
                        activationBkgColor: '#34495e',
                        sequenceNumberColor: '#ffffff',

                        // Gantt 甘特图
                        sectionBkgColor: '#243447',
                        altSectionBkgColor: '#1a252f',
                        sectionBkgColor2: '#34495e',
                        excludeBkgColor: '#1a252f',
                        taskBorderColor: '#5dade2',
                        taskBkgColor: '#2980b9',
                        taskTextColor: '#ffffff',
                        taskTextOutsideColor: '#ffffff',
                        taskTextLightColor: '#ffffff',
                        taskTextDarkColor: '#ffffff',
                        taskTextClickableColor: '#ffffff',
                        activeTaskBkgColor: '#2980b9',
                        activeTaskBorderColor: '#5dade2',
                        doneTaskBkgColor: '#27ae60',
                        doneTaskBorderColor: '#58d68d',
                        critBkgColor: '#c0392b',
                        critBorderColor: '#e74c3c',
                        gridColor: '#34495e',
                        todayLineColor: '#e74c3c',
                        section0: '#2980b9',
                        section1: '#27ae60',
                        section2: '#8e44ad',
                        section3: '#c0392b',

                        // State / Journey / Flowchart 通用
                        stateBkg: '#2980b9',
                        stateLabelColor: '#ffffff',
                        transitionColor: '#ffffff',
                        transitionLabelColor: '#ffffff',
                        personBorder: '#5dade2',
                        personBkg: '#2980b9',
                        compositeBorder: '#3498db',
                        compositeBackground: '#243447',
                        altBackground: '#1a252f',
                        compositeTitleBackground: '#34495e',
                        innerEndBackground: '#1a252f',
                        specialStateColor: '#f39c12',
                        errorBkgColor: '#c0392b',
                        errorTextColor: '#ffffff',

                        // Mindmap / Timeline / Pie 等色阶（cScale0-11）
                        cScale0: '#2980b9',
                        cScale1: '#27ae60',
                        cScale2: '#8e44ad',
                        cScale3: '#c0392b',
                        cScale4: '#d35400',
                        cScale5: '#16a085',
                        cScale6: '#f39c12',
                        cScale7: '#3498db',
                        cScale8: '#9b59b6',
                        cScale9: '#1abc9c',
                        cScale10: '#e67e22',
                        cScale11: '#2ecc71',
                        scaleLabelColor: '#ffffff',

                        // Class / ER 图
                        classText: '#ffffff',
                        relationColor: '#ffffff',
                        relationLabelColor: '#ffffff',
                        relationLabelBackground: '#1a2332',
                        attributeBackgroundColorOdd: '#34495e',
                        attributeBackgroundColorEven: '#2c3e50',

                        // Requirement Diagram
                        requirementBackground: '#243447',
                        requirementBorderColor: '#3498db',
                        requirementBorderSize: '1px',
                        requirementTextColor: '#ffffff',

                        // Pie Chart
                        pieTitleTextColor: '#ffffff',
                        pieSectionTextColor: '#ffffff',
                        pieLegendTextColor: '#ffffff',
                        pieStrokeColor: '#1a2332',
                        pieOuterStrokeColor: '#ffffff',

                        // Quadrant Chart
                        quadrant1Fill: '#2980b9',
                        quadrant2Fill: '#27ae60',
                        quadrant3Fill: '#8e44ad',
                        quadrant4Fill: '#c0392b',
                        quadrant1TextFill: '#ffffff',
                        quadrant2TextFill: '#ffffff',
                        quadrant3TextFill: '#ffffff',
                        quadrant4TextFill: '#ffffff',
                        quadrantPointFill: '#f39c12',
                        quadrantPointTextFill: '#ffffff',
                        quadrantXAxisTextFill: '#ffffff',
                        quadrantYAxisTextFill: '#ffffff',
                        quadrantInternalBorderStrokeFill: '#34495e',
                        quadrantExternalBorderStrokeFill: '#5dade2',
                        quadrantTitleFill: '#ffffff',

                        // Gitgraph
                        git0: '#2980b9',
                        git1: '#27ae60',
                        git2: '#8e44ad',
                        git3: '#c0392b',
                        git4: '#d35400',
                        git5: '#16a085',
                        git6: '#f39c12',
                        git7: '#3498db',
                        gitInv0: '#34495e',
                        gitInv1: '#2c3e50',
                        gitInv2: '#1a252f',
                        gitInv3: '#243447',
                        gitInv4: '#1a252f',
                        gitInv5: '#2c3e50',
                        gitInv6: '#34495e',
                        gitInv7: '#243447',
                        branchLabelColor: '#ffffff',
                        gitBranchLabel0: '#ffffff',
                        gitBranchLabel1: '#ffffff',
                        gitBranchLabel2: '#ffffff',
                        gitBranchLabel3: '#ffffff',
                        gitBranchLabel4: '#ffffff',
                        gitBranchLabel5: '#ffffff',
                        gitBranchLabel6: '#ffffff',
                        gitBranchLabel7: '#ffffff',
                        tagLabelColor: '#ffffff',
                        tagLabelBackground: '#8e44ad',
                        tagLabelBorder: '#af7ac5',
                        commitLabelColor: '#ffffff',
                        commitLabelBackground: '#2c3e50'
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
