"""
Export the deployed triage model to ONNX for the IBM Z AI stack.

The graph uses only core ONNX ops (Sub, Div, MatMul, Add, Softmax), no
ai.onnx.ml ops, so it can be compiled by the IBM Z Deep Learning Compiler
(zDLC / onnx-mlir). MatMul, Add and Softmax are among the ops zDLC can
offload to the Telum Integrated Accelerator for AI (NNPA) on IBM z16+ /
LinuxONE 4+. The pure-Python JSON engine stays as the portable fallback.

    python scripts/export_onnx.py          # writes model/triage_model.onnx

    # On LinuxONE / IBM Z with podman or docker (image from the IBM Z container registry):
    # podman run --rm -v $PWD/model:/workdir:z icr.io/ibmz_zdlc/onnx-mlir:<version> \
    #     --EmitLib --O3 --mcpu=z16 --mtriple=s390x-ibm-loz --maccel=NNPA triage_model.onnx

Also verifies with ONNX Runtime that the ONNX output matches the JSON engine.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


def build():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = json.load(open(os.path.join(ROOT, "model", "triage_model.json")))
    n, k = len(m["features"]), len(m["labels"])
    inits = [
        numpy_helper.from_array(np.array(m["scaler_mean"], dtype=np.float32), "mean"),
        numpy_helper.from_array(np.array(m["scaler_scale"], dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array(m["coef"], dtype=np.float32).T.copy(), "W"),
        numpy_helper.from_array(np.array(m["intercept"], dtype=np.float32), "b"),
    ]
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["scaled"]),
        helper.make_node("MatMul", ["scaled", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "pavan_sewa_triage",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", n])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["N", k])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="pavan-sewa-command")
    model.ir_version = 8
    onnx.checker.check_model(model)
    out = os.path.join(ROOT, "model", "triage_model.onnx")
    onnx.save(model, out)
    print("wrote", out)
    return out


def verify(path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipped parity check")
        return
    import pandas as pd
    from engine.features import engineer
    from engine.triage import TriageModel
    df = pd.read_csv(os.path.join(ROOT, "data", "incidents.csv")).head(500)
    recs = df.to_dict("records")
    X = np.array([engineer(r) for r in recs], dtype=np.float32)
    probs = ort.InferenceSession(path).run(None, {"features": X})[0]
    tm = TriageModel()
    ref = np.array([[tm.predict(r)["probabilities"][lbl] for lbl in tm.labels] for r in recs])
    print("max |ONNX - JSON engine| =", float(np.abs(probs - ref).max()))


if __name__ == "__main__":
    verify(build())
