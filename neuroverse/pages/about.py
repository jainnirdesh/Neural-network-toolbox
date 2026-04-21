from neuroverse.components import page_header, render_use_cases, text_card


def render_about_page() -> None:
    page_header("About", "NeuroVerse AI Lab project information")
    text_card(
        "Project Overview",
        """
        NeuroVerse AI Lab demonstrates machine learning concepts with interactive controls, visual outputs,
        and short educational content for each module.
        """,
    )

    text_card(
        "Learning Goals",
        """
        <ul>
            <li>Understand the difference between classification and regression tasks.</li>
            <li>Interpret model outputs and confidence scores.</li>
            <li>Build intuition for neural network structure and data flow.</li>
            <li>Connect concepts to practical real-world applications.</li>
        </ul>
        """,
    )

    render_use_cases([
        "Education",
        "Prototype Demos",
        "AI Workshops",
        "Concept Validation",
    ])
