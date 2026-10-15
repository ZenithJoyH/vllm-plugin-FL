"""GDN mode routing and stateful error contracts after the MetaX ABI update."""
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize("available", [True, False])
def test_mtp_and_non_mtp_routing_idempotence(monkeypatch, available):
    from vllm.model_executor.layers.mamba import gdn_linear_attn as mod
    from vllm_fl.patches import gdn_linear_attn as hook
    events=[]
    original_method=mod.GatedDeltaNetAttention._forward_core_decode_non_spec
    native=lambda **kw: events.append(("native", kw)) or "native"
    monkeypatch.setattr(mod,"fla_chunk_gated_delta_rule",native)
    monkeypatch.setattr(mod,"fused_sigmoid_gating_delta_rule_update",native)
    monkeypatch.setattr(mod,"fused_recurrent_gated_delta_rule_packed_decode",native)
    monkeypatch.setattr(mod,"_flagos_gdn_installed",False,raising=False)
    registry=SimpleNamespace(get_implementations=lambda name:[SimpleNamespace(is_available=lambda:available)])
    manager=SimpleNamespace(registry=registry,ensure_initialized=lambda:None)
    monkeypatch.setattr(hook,"get_default_manager",lambda:manager)
    def dispatch(name,**kw):
        events.append((name,kw));return "fg"
    monkeypatch.setattr(hook,"call_op",dispatch)
    hook.install_gdn_dispatch();first=mod.fused_recurrent_gated_delta_rule_packed_decode
    hook.install_gdn_dispatch();assert mod.fused_recurrent_gated_delta_rule_packed_decode is first
    assert mod.GatedDeltaNetAttention._forward_core_decode_non_spec is original_method
    accepted=torch.tensor([2,1]);indices=torch.tensor([[1,2,3],[4,5,6]])
    assert mod.fused_sigmoid_gating_delta_rule_update(1,2,3,4,5,6,7,num_accepted_tokens=accepted,ssm_state_indices=indices)==("fg" if available else "native")
    assert events[-1][0]==("gdn_decode" if available else "native")
    assert events[-1][1]["num_accepted_tokens"] is accepted
    assert events[-1][1]["ssm_state_indices"] is indices
    assert first(1,2,3,4,5,6,7,8,9)==("fg" if available else "native")
    assert events[-1][0]==("gdn_packed_decode" if available else "native")
    # Errors after partial state updates must not trigger a native retry.
    if available:
        before=len(events)
        def fail(name,**kw):events.append((name,kw));raise RuntimeError("partial update")
        monkeypatch.setattr(hook,"call_op",fail)
        with pytest.raises(RuntimeError,match="partial update"):first(1,2,3,4,5,6,7,8,9)
        assert len(events)==before+1
    # KDA does not silently enter a GDN kernel.
    mod.fused_sigmoid_gating_delta_rule_update(1,2,3,4,5,6,7,is_kda=True)
    assert events[-1][0]=="native"

def test_backend_fwd_preserves_v_major_and_spec_indices(monkeypatch):
    import flaggems_vllm
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    state=torch.randn(8,4,48,32);indices=torch.tensor([[1,2,3],[4,5,6]]);accepted=torch.tensor([2,1]);seen={}
    def fwd(**kw):seen.update(kw);return kw["q"],kw["initial_state"]
    monkeypatch.setattr(flaggems_vllm,"fused_recurrent_gated_delta_rule_fwd",fwd)
    q=torch.randn(1,6,2,32);v=torch.randn(1,6,4,48)
    _,out=FlagGemsBackend().gdn_decode(torch.zeros(4),torch.zeros(6,4),torch.zeros(6,4),torch.zeros(4),q,q,v,initial_state=state,ssm_state_indices=indices,num_accepted_tokens=accepted)
    assert out is state and seen["initial_state"] is state
    assert seen["num_accepted_tokens"] is accepted and seen["ssm_state_indices"] is indices

@pytest.mark.parametrize("vendor", ["metax", "nvidia"])
def test_prefill_converts_only_gathered_state_layout(monkeypatch, vendor):
    import flaggems_vllm
    monkeypatch.setattr(flaggems_vllm,"vendor_name",vendor)
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    backend=FlagGemsBackend();state=torch.arange(2*4*48*32).view(2,4,48,32).float();seen={}
    def chunk(**kw):
        seen.update(kw);return None,kw["v"],None,kw["initial_state"]+1,None,None,None
    monkeypatch.setattr(backend,"chunk_gated_delta_rule_fwd",chunk)
    q=torch.zeros(1,16,2,32);v=torch.zeros(1,16,4,48)
    _,final=backend.gdn_prefill(q,q,v,torch.zeros(1,16,4),torch.zeros(1,16,4),initial_state=state,output_final_state=True)
    assert seen["initial_state"].shape==((2,4,48,32) if vendor=="metax" else (2,4,32,48))
    torch.testing.assert_close(seen["initial_state"],state if vendor=="metax" else state.transpose(-1,-2))
    torch.testing.assert_close(final,state+1)
