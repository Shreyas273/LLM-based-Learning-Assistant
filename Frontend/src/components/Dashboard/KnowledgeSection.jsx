import React, { useEffect, useState, useMemo } from 'react';
import { getPdfIndex, comparativeAnalysis, getKnowledgeGraph } from '../../services/api';
import { motion } from 'framer-motion';
import { Loader2, FileText, Network, Search, ArrowRight, Database, AlertCircle } from 'lucide-react';
import ReactFlow, { Background, Controls, MiniMap, MarkerType } from "reactflow";
import "reactflow/dist/style.css";
import dagre from "dagre";

const KnowledgeSection = () => {
    const [pdfIndex, setPdfIndex] = useState(null);
    const [loading, setLoading] = useState(true);
    const [analysisTopic, setAnalysisTopic] = useState('');
    const [analysisResult, setAnalysisResult] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [graphData, setGraphData] = useState(null);
    const [generatingGraph, setGeneratingGraph] = useState(false);

    useEffect(() => {
        loadPdfIndex();
    }, []);


    const loadPdfIndex = async () => {
        try {
            const response = await getPdfIndex();
            setPdfIndex(response);
        } catch (error) {
            console.error("Failed to load PDF index:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleAnalyze = async (e) => {
        e.preventDefault();
        if (!analysisTopic.trim()) return;

        setAnalyzing(true);
        setAnalysisResult(null);
        try {
            const response = await comparativeAnalysis({
                topic: analysisTopic,
                include_all_pdfs: true
            });
            setAnalysisResult(response);
        } catch (error) {
            console.error("Analysis failed:", error);
            setAnalysisResult({ error: "Failed to perform analysis. Please try again." });
        } finally {
            setAnalyzing(false);
        }
    };

    const handleGenerateGraph = async () => {
        setGeneratingGraph(true);
        setGraphData(null);
        try {
            const response = await getKnowledgeGraph();
            const graph = response?.graph || response;
            const layouted = buildGraph(graph);
            if (layouted) setGraphData(layouted);
        } catch (error) {
            console.error("Graph generation failed:", error);
        } finally {
            setGeneratingGraph(false);
        }
    };

    const pdfList = useMemo(() => {
        if (!pdfIndex?.pdfs || Object.keys(pdfIndex.pdfs).length === 0) {
            return <p className="text-slate-500 text-sm">No documents indexed yet.</p>;
        }

        return (
            <div className="space-y-3">
                <div className="text-xs text-slate-400 mb-2 flex justify-between">
                    <span>{pdfIndex.total_pdfs} Documents</span>
                    <span>{pdfIndex.total_topics} Topics Extracted</span>
                </div>
                {Object.entries(pdfIndex.pdfs).map(([name, info]) => (
                    <div key={name} className="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                        <div className="flex items-start justify-between">
                            <div className="flex items-center space-x-2 overflow-hidden">
                                <FileText size={16} className="text-slate-400 flex-shrink-0" />
                                <span className="text-sm font-medium text-slate-200 truncate" title={name}>{name}</span>
                            </div>
                            <span className="text-xs text-slate-500 flex-shrink-0">{info.chunk_count} chunks</span>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-1">
                            {info.topics.slice(0, 3).map((topic, i) => (
                                <span key={i} className="text-[10px] bg-slate-700/50 text-slate-300 px-1.5 py-0.5 rounded">
                                    {topic}
                                </span>
                            ))}
                            {info.topics.length > 3 && (
                                <span className="text-[10px] text-slate-500 px-1">+{info.topics.length - 3}</span>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        );
    }, [pdfIndex]);

    if (loading) return <div className="flex h-64 items-center justify-center"><Loader2 className="animate-spin text-sky-500" /></div>;

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* PDF Index List */}
                <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="glass-panel p-6 rounded-2xl h-fit"
                >
                    <h3 className="text-xl font-bold text-slate-100 mb-4 flex items-center">
                        <Database className="mr-2 text-blue-400" size={20} />
                        Knowledge Base
                    </h3>

                    {pdfList}
                </motion.div>

                {/* Comparative Analysis */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-panel p-6 rounded-2xl lg:col-span-2 flex flex-col"
                >
                    <h3 className="text-xl font-bold text-slate-100 mb-4 flex items-center">
                        <Search className="mr-2 text-purple-400" size={20} />
                        Deep Analysis
                    </h3>

                    <form onSubmit={handleAnalyze} className="flex gap-2 mb-6">
                        <input
                            type="text"
                            value={analysisTopic}
                            onChange={(e) => setAnalysisTopic(e.target.value)}
                            placeholder="Enter a complex topic to analyze across all documents..."
                            className="flex-1 bg-slate-950/50 border border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-200 focus:ring-1 focus:ring-purple-500 outline-none"
                        />
                        <button
                            type="submit"
                            disabled={analyzing || !analysisTopic.trim()}
                            className="bg-purple-600 hover:bg-purple-500 text-white px-4 py-2 rounded-xl text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                        >
                            {analyzing ? <Loader2 className="animate-spin" size={18} /> : 'Analyze'}
                        </button>
                    </form>

                    {analysisResult && (
                        <div className="flex-1 bg-slate-900/40 rounded-xl p-4 border border-slate-700/30 overflow-y-auto max-h-[400px] custom-scrollbar">
                            {analysisResult.error ? (
                                <div className="text-rose-400 flex items-center"><AlertCircle className="mr-2" size={16} /> {analysisResult.error}</div>
                            ) : (
                                <div className="space-y-4">
                                    <div dangerouslySetInnerHTML={{ __html: analysisResult.analysis?.replace(/\n/g, '<br/>') }} className="prose prose-invert prose-sm max-w-none text-slate-300" />

                                    {/* Covered PDFs (if returned by API) */}
                                    {analysisResult.covered_pdfs && (
                                        <div className="mt-4 pt-4 border-t border-slate-700/50">
                                            <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">Covered PDFs</h4>
                                            <div className="flex flex-wrap gap-2">
                                                {analysisResult.covered_pdfs.map((p, idx) => (
                                                    <span key={idx} className="text-[11px] bg-slate-800/50 border border-slate-700/50 text-slate-300 px-2 py-1 rounded-lg">
                                                        {p}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Sources if available in the structure */}
                                    {analysisResult.sources && (
                                        <div className="mt-4 pt-4 border-t border-slate-700/50">
                                            <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">Sources</h4>
                                            {analysisResult.sources.map((src, idx) => (
                                                <div key={idx} className="text-xs text-slate-400 mb-1">
                                                    • {src}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {!analysisResult && !analyzing && (
                        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 py-12">
                            <Search size={48} className="opacity-20 mb-4" />
                            <p className="text-sm">Search for a topic to see how it connects across your PDFs.</p>
                        </div>
                    )}
                </motion.div>
            </div>

            {/* Visual Knowledge Graph Placeholder/Implementation */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-panel p-6 rounded-2xl"
            >
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold text-slate-100 flex items-center">
                        <Network className="mr-2 text-pink-400" size={20} />
                        Concept Map
                    </h3>
                    <button
                        onClick={handleGenerateGraph}
                        disabled={generatingGraph}
                        className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors flex items-center"
                    >
                        {generatingGraph ? <Loader2 className="animate-spin mr-2" size={12} /> : null}
                        Regenerate Graph
                    </button>
                </div>

                <div className="bg-slate-950/50 rounded-xl border border-slate-800 h-[400px]">

                    {graphData?.nodes ? (

                        <ReactFlow
                            nodes={graphData.nodes}
                            edges={graphData.edges}
                            fitView
                        >
                            <MiniMap />
                            <Controls />
                            <Background gap={16} />
                        </ReactFlow>

                    ) : (

                        <div className="flex h-full items-center justify-center text-slate-500">
                            Click "Regenerate Graph" to visualize knowledge connections.
                        </div>

                    )}

                </div>

            </motion.div>
        </div>
    );
};

const getLayoutedElements = (nodes, edges) => {
    // Dagre layout configuration tuned for clearer, more compact graphs
    // Hierarchical layout: subject (top) → topic → subtopic (top-to-bottom).
    const nodeWidth = 260;
    const nodeHeight = 60;
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({
        rankdir: "TB",
        ranksep: 90,
        nodesep: 40,
        marginx: 80,
        marginy: 40,
    });

    nodes.forEach((node) => {
        g.setNode(node.id, { width: nodeWidth, height: nodeHeight });
    });

    edges.forEach((edge) => {
        g.setEdge(edge.source, edge.target);
    });

    dagre.layout(g);

    const layoutedNodes = nodes.map((node) => {
        const pos = g.node(node.id);
        // Center React Flow nodes on dagre coordinates
        return {
            ...node,
            position: { x: pos.x - nodeWidth / 2, y: pos.y - nodeHeight / 2 },
        };
    });

    return { nodes: layoutedNodes, edges };
};

const buildGraph = (graph) => {
    if (!graph?.nodes?.length) return null;

    const nodeIds = new Set();
    const nodes = graph.nodes.map((n) => {
        const id = String(n.id).replace(/[^a-zA-Z0-9_-]/g, '_');
        nodeIds.add(id);
        const type = n.type || "topic";
        const isSubject = type === "subject";
        const isTopic = type === "topic";
        const isSubtopic = type === "subtopic";

        const background = isSubject
            ? "rgba(2,132,199,0.95)"   // sky-600
            : isTopic
                ? "rgba(15,23,42,0.95)" // slate-900
                : "rgba(2,6,23,0.95)";  // slate-950
        const border = isSubject ? "#38bdf8" : isTopic ? "#334155" : "#1e293b";

        return {
            id,
            data: { label: n.label || n.id },
            position: { x: 0, y: 0 },
            style: {
                padding: 12,
                borderRadius: 12,
                fontSize: "12px",
                fontWeight: 500,
                background,
                color: "white",
                border: `1px solid ${border}`,
                boxShadow: isSubject
                    ? "0 14px 36px rgba(56,189,248,0.35)"
                    : "0 10px 28px rgba(15,23,42,0.65)",
            },
        };
    });

    const idMap = {};
    graph.nodes.forEach((n, i) => {
        const safeId = String(n.id).replace(/[^a-zA-Z0-9_-]/g, '_');
        idMap[n.id] = safeId;
    });

    const edges = (graph.edges || []).map((e, i) => {
        const source = idMap[e.from] || idMap[e.source] || String(e.from || e.source).replace(/[^a-zA-Z0-9_-]/g, '_');
        const target = idMap[e.to] || idMap[e.target] || String(e.to || e.target).replace(/[^a-zA-Z0-9_-]/g, '_');
        if (!nodeIds.has(source) || !nodeIds.has(target)) return null;
        return {
            id: `edge-${i}`,
            source,
            target,
            label: e.label || e.relation || "",
            type: "smoothstep",
            animated: false,
            style: { stroke: "#38bdf8" },
            markerEnd: {
                type: MarkerType.ArrowClosed,
                color: "#38bdf8",
            },
            labelBgPadding: [6, 2],
            labelBgBorderRadius: 4,
            labelBgStyle: {
                fill: "rgba(15,23,42,0.95)",
                stroke: "#020617",
                strokeWidth: 0.5,
            },
        };
    }).filter(Boolean);

    return getLayoutedElements(nodes, edges);
};

export default KnowledgeSection;
