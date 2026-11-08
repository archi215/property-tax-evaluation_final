"""Headless smoke test: every page renders without an exception (uses Streamlit's AppTest)."""
import os
import pytest
import config

pytestmark = pytest.mark.skipif(not os.path.exists(config.METADATA_JSON), reason="model not trained")
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODS = "from views import bulk, data_pages, finance_pages, market_pages, simulator, studio, trust\n"
PAGES = ["studio.render()", "simulator.render()", "bulk.render_bulk()", "finance_pages.render_emi()", "finance_pages.render_afford()",
         "finance_pages.render_rentbuy()", "finance_pages.render_investor()", "market_pages.render_map()", "market_pages.render_faceoff()",
         "trust.render_health()", "trust.render_conformal()", "trust.render_how()", "data_pages.render_rules()"]


def _run(code):
    return AppTest.from_string(f"import sys; sys.path.insert(0, {ROOT!r})\nimport theme\ntheme.inject()\n{MODS}{code}", default_timeout=180).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    assert not _run(page).exception


def test_home_and_navigation():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception


def test_studio_is_live():
    at = _run("studio.render()")
    at.slider(key="s_area").set_value(2000).run()
    at.selectbox(key="s_state").set_value("Maharashtra").run()
    assert not at.exception and at.session_state["s_city"] == "Mumbai"
    assert at.session_state["studio_result"]["inputs"]["built_up_area_sqft"] == 2000
