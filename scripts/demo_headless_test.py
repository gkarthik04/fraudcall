"""Run the Streamlit app headlessly (streamlit.testing) on a prepared sample and an uploaded-style flow; record timing."""
import json, time
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("src/demo/app.py", default_timeout=600); at.run()
assert not at.exception, at.exception
res = {}
for sid in ("sample_1", "sample_6"):
    at.sidebar.selectbox[0].set_value(sid)
    at.sidebar.selectbox[1].set_value("GSM")
    t = time.time(); at.sidebar.button[0].click().run(); el = time.time() - t
    assert not at.exception, at.exception
    res[sid] = {"seconds": round(el, 1), "metrics": [(m.label, m.value) for m in at.metric], "info": [i.value for i in at.info]}
    print(sid, res[sid], flush=True)
json.dump(res, open("results/demo_headless.json", "w"), indent=1)
