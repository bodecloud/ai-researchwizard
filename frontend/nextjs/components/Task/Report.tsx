import React, { useEffect, useState } from 'react';
import { markdownToHtml } from '../../helpers/markdownHelper';
import '../../styles/markdown.css';

interface ReportProps {
    report: string;
}

export default function Report({ report }: ReportProps) {
    const [htmlContent, setHtmlContent] = useState('');

    useEffect(() => {
        const convertMarkdownToHtml = async () => {
            try {
                const processedHtml = await markdownToHtml(report);
                setHtmlContent(processedHtml);
            } catch (error) {
                console.error('Error converting markdown to HTML:', error);
                setHtmlContent('<p>Error rendering content</p>');
            }
        };

        if (report) {
            convertMarkdownToHtml();
        }
    }, [report]);

    return (
        <div>
            <h2>Research Report</h2>
            <div id="reportContainer" className="markdown-content">
                <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
            </div>
        </div>
    );
};