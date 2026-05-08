# Python DCT Transform Coding Demo

This is a Python web demo for visualizing JPEG-style DCT transform coding.
It keeps the core flow compact:

- load a cloud-side example image from `examples/`;
- convert RGB to YCbCr;
- run 8x8 DCT, JPEG-style quantisation, coefficient masking, dequantisation and IDCT;
- click DCT basis tiles to enable or remove coefficients;
- click the input image to inspect the corresponding 8x8 Y-channel block tables.

## Run

From the repository root:

```bash
python3 -m pip install -r python_dct_demo/requirements.txt
python3 python_dct_demo/app.py --port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

The demo uses only Python's standard-library HTTP server plus `numpy` and
`Pillow`.

## Cloud Studio

Cloud Studio preview needs the server to listen on `0.0.0.0`:

```bash
python3 -m pip install -r python_dct_demo/requirements.txt
python3 python_dct_demo/app.py --host 0.0.0.0 --port 8501
```

The repository includes `.vscode/preview.yml`, so Cloud Studio can run the demo
from the Run button and open the port preview for `8501`.
