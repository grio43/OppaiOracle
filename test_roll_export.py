
import torch
import torch.nn as nn
import onnx

class RollModel(nn.Module):
    def forward(self, x):
        # Simulate SwinV2 cyclic shift: roll by -shift_size
        return torch.roll(x, shifts=(-1, -1), dims=(1, 2))

def test_export():
    model = RollModel()
    dummy_input = torch.randn(1, 4, 4, 3) # B, H, W, C
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "test_roll.onnx",
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            verbose=False
        )
        print("SUCCESS: torch.roll exported with opset 18!")
        
        # Verify the model
        onnx_model = onnx.load("test_roll.onnx")
        onnx.checker.check_model(onnx_model)
        print("SUCCESS: Exported model is valid ONNX.")
        return True
    except Exception as e:
        print(f"FAILURE: Export failed: {e}")
        return False

if __name__ == "__main__":
    test_export()
