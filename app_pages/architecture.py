from pathlib import Path

import streamlit as st

DOCS = Path(__file__).resolve().parent.parent / "docs"

st.title("Architecture & documentation", anchor=False)
st.caption("These pages render the Markdown files in the docs/ folder, so the app and the repository stay in sync.")

tabs = st.tabs([":material/account_tree: Architecture", ":material/menu_book: Methodology", ":material/analytics: Evaluation"])
for tab, name in zip(tabs, ["architecture.md", "methodology.md", "evaluation.md"]):
    with tab:
        path = DOCS / name
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.warning(f"docs/{name} is missing.")
