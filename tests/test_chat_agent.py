from app.services.chat_agent import ChatAgentService


def test_search_summary_lists_multiple_prefix_matches():
    summary = ChatAgentService._format_search_summary(
        "N",
        [
            {"display": "Nesh Test"},
            {"display": "Nora Lane"},
            {"display": "Noah Smith"},
        ],
        search_mode="starts_with",
    )

    assert "names start with 'N'" in summary
    assert "Nesh Test" in summary
    assert "Nora Lane" in summary
    assert "Noah Smith" in summary


def test_search_summary_handles_no_matches():
    summary = ChatAgentService._format_search_summary("N", [], search_mode="starts_with")

    assert summary == "I could not find any patients whose name starts with 'N'."
