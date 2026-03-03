"""Report generator skill for GPT Researcher.

This module provides the ReportGenerator class that handles report
writing, including introductions, conclusions, and subtopic management.
"""

import json
from typing import Any, Dict, Optional

from ..actions import (
    generate_draft_section_titles,
    generate_report,
    stream_output,
    write_conclusion,
    write_report_introduction,
)
from ..utils.llm import construct_subtopics


class ReportGenerator:
    """Generates reports based on research data.

    This class handles all aspects of report generation including
    writing introductions, conclusions, and managing report structure.

    Attributes:
        researcher: The parent GPTResearcher instance.
        research_params: Dictionary of parameters for report generation.
    """

    def __init__(self, researcher):
        """Initialize the ReportGenerator.

        Args:
            researcher: The GPTResearcher instance that owns this generator.
        """
        self.researcher = researcher
        self.research_params = {
            "query": self.researcher.query,
            "agent_role_prompt": (
                self.researcher.cfg.agent_role  # pyright: ignore[reportAttributeAccessIssue]
                or self.researcher.role
            ),
            "report_type": self.researcher.report_type,
            "report_source": self.researcher.report_source,
            "tone": self.researcher.tone,
            "websocket": self.researcher.websocket,
            "cfg": self.researcher.cfg,
            "headers": self.researcher.headers,
        }

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None, custom_prompt="", available_images: list = None) -> str:
        """
        Write a report based on existing headers and relevant contents.

        Args:
            existing_headers (list): List of existing headers.
            relevant_written_contents (list): List of relevant written contents.
            ext_context (Optional): External context, if any.
            custom_prompt (str): Custom prompt for the report.
            available_images (list): Pre-generated images available for embedding.

        Returns:
            str: The generated report.
        """
        available_images = available_images or []
        
        # send the selected images prior to writing report
        research_images = self.researcher.get_research_images()
        if research_images:
            await stream_output(
                "images",
                "selected_images",
                json.dumps(research_images),
                self.researcher.websocket,
                True,
                research_images
            )

        context = ext_context or self.researcher.context
        
        # Log image availability
        if available_images and self.researcher.verbose:
            await stream_output(
                "logs",
                "images_available",
                f"🖼️ {len(available_images)} pre-generated images available for embedding",
                self.researcher.websocket,
            )
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_report",
                f"✍️ Writing report for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        report_params = self.research_params.copy()
        report_params["context"] = context
        report_params["custom_prompt"] = custom_prompt
        report_params["available_images"] = available_images  # Pass pre-generated images

        if should_use_rag:
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "using_rag_generation",
                    "🧠 Using RAG-based generation for comprehensive report...",
                    self.researcher.websocket,
                )

            # Use the new RAG-based report generation
            from gpt_researcher.actions.report_generation import generate_report_with_rag

            report: str = await generate_report_with_rag(
                query=self.researcher.query,
                context=context,
                agent_role_prompt=(
                    self.researcher.cfg.agent_role  # pyright: ignore[reportAttributeAccessIssue]
                    or self.researcher.role
                ),
                report_type=self.researcher.report_type,
                tone=self.researcher.tone,
                report_source=self.researcher.report_source,
                websocket=self.researcher.websocket,
                cfg=self.researcher.cfg,
                memory=self.researcher.memory,  # Pass memory for RAG
                main_topic=getattr(self.researcher, 'parent_query', ''),
                existing_headers=existing_headers,
                relevant_written_contents=relevant_written_contents,
                cost_callback=self.researcher.add_costs,
                custom_prompt=custom_prompt,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family,
            )
        else:
            # Use original method for smaller contexts
            report_params: dict[str, Any] = self.research_params.copy()
            report_params["context"] = context

            if self.researcher.report_type == "subtopic_report":
                report_params.update(
                    {
                        "main_topic": self.researcher.parent_query,
                        "existing_headers": existing_headers,
                        "relevant_written_contents": relevant_written_contents,
                        "cost_callback": self.researcher.add_costs,
                        "custom_prompt": custom_prompt,
                    }
                )
            else:
                report_params["cost_callback"] = self.researcher.add_costs

            report = await generate_report(**report_params, **self.researcher.kwargs)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "report_written",
                f"📝 Report written for '{self.researcher.query}' ({len(report)} characters)",
                self.researcher.websocket,
            )

        # Finish visualization and generate flow diagram
        if visualizer.is_enabled():
            mermaid_diagram: str | None = visualizer.finish_flow()

            # Stream the visual diagram to the frontend if websocket available
            if mermaid_diagram and self.researcher.websocket:
                await stream_output(
                    "logs",
                    "llm_flow_diagram",
                    f"🎨 LLM Interaction Flow:\n{mermaid_diagram}",
                    self.researcher.websocket,
                )

        return report

    def _should_use_rag_generation(
        self,
        context: list[dict[str, Any]],
    ) -> bool:
        """Determine if RAG-based generation should be used based on context size.

        Args:
            context: The research context

        Returns:
            bool: True if RAG should be used, False otherwise
        """
        if not context:
            return False

        # Calculate total context size
        total_chars: int = 0
        total_items: int = len(context)

        for item in context:
            if isinstance(item, dict):
                if 'raw_content' in item:
                    total_chars += len(str(item['raw_content']))
                elif 'content' in item:
                    total_chars += len(str(item['content']))
                else:
                    total_chars += len(str(item))
            else:
                total_chars += len(str(item))

        # Use RAG if:
        # 1. We have a lot of research items (>20)
        # 2. OR total content is large (>50k characters)
        # 3. OR we're doing a detailed report
        use_rag: bool = (
            total_items > 20 or
            total_chars > 50000 or
            self.researcher.report_type == "detailed_report"
        )

        if use_rag and self.researcher.verbose:
            asyncio.create_task(stream_output(
                "logs",
                "rag_decision",
                f"🔍 RAG enabled: {total_items} items, {total_chars:,} chars, type: {self.researcher.report_type}",
                self.researcher.websocket,
            ))

        return use_rag

    async def write_report_conclusion(
        self,
        report_content: str,
    ) -> str:
        """Write the conclusion for the report.

        Args:
            report_content (str): The content of the report.

        Returns:
            str: The generated conclusion.
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_conclusion",
                f"✍️ Writing conclusion for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        conclusion: str = await write_conclusion(
            query=self.researcher.query,
            context=report_content,
            config=self.researcher.cfg,
            agent_role_prompt=(
                self.researcher.cfg.agent_role  # pyright: ignore[reportAttributeAccessIssue]
                or self.researcher.role
            ),
            cost_callback=self.researcher.add_costs,
            websocket=self.researcher.websocket,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "conclusion_written",
                f"📝 Conclusion written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return conclusion

    async def write_introduction(self) -> str:
        """Write the introduction section of the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_introduction",
                f"✍️ Writing introduction for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        introduction: str = await write_report_introduction(
            query=self.researcher.query,
            context=self.researcher.context,
            agent_role_prompt=(
                self.researcher.cfg.agent_role  # pyright: ignore[reportAttributeAccessIssue]
                or self.researcher.role
            ),
            config=self.researcher.cfg,
            websocket=self.researcher.websocket,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "introduction_written",
                f"📝 Introduction written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return introduction

    async def get_subtopics(self) -> list[str]:
        """Retrieve subtopics for the research."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_subtopics",
                f"🌳 Generating subtopics for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        subtopics: list[str] = await construct_subtopics(
            task=self.researcher.query,
            data=self.researcher.context,
            config=self.researcher.cfg,
            subtopics=self.researcher.subtopics,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subtopics_generated",
                f"📊 Subtopics generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return subtopics

    async def get_draft_section_titles(
        self,
        current_subtopic: str,
    ) -> list[str]:
        """Generate draft section titles for the report.

        Args:
            current_subtopic (str): The current subtopic.

        Returns:
            list[str]: The draft section titles.
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_draft_sections",
                f"📑 Generating draft section titles for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        draft_section_titles: list[str] = await generate_draft_section_titles(
            query=self.researcher.query,
            current_subtopic=current_subtopic,
            context=self.researcher.context,
            role=(
                self.researcher.cfg.agent_role  # pyright: ignore[reportAttributeAccessIssue]
                or self.researcher.role
            ),
            websocket=self.researcher.websocket,
            config=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "draft_sections_generated",
                f"🗂️ Draft section titles generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return draft_section_titles
