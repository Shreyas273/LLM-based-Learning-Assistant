import React, { useState } from 'react';
import { TOOLS } from './constants';
import ToolCard from './ToolCard';
import ToolModal from './ToolModal';

const ToolsSection = () => {
    const [selectedTool, setSelectedTool] = useState(null);

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="text-center space-y-4">
                <h2 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-fuchsia-400">
                    AI Power Tools
                </h2>
                <p className="text-slate-400 max-w-2xl mx-auto">
                    Supercharge your learning with our suite of specialized AI tools.
                    Summarize documents, generate content, and solve complex problems in seconds.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {TOOLS.map((tool) => (
                    <ToolCard
                        key={tool.id}
                        tool={tool}
                        onClick={setSelectedTool}
                    />
                ))}
            </div>

            {selectedTool && (
                <ToolModal
                    tool={selectedTool}
                    onClose={() => setSelectedTool(null)}
                />
            )}
        </div>
    );
};

export default ToolsSection;
