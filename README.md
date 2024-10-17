# VaePaster

VaePaster is a powerful tool for transforming and aligning images with text labels. It provides both conditional and unconditional alignment capabilities, making it versatile for various image processing tasks.

## 🚀 Features

- Conditional image alignment
- Unconditional image alignment
- Text detection and recognition using PaddleOCR
- Customizable settings via YAML configuration

## 📁 Project Structure
vaepaster/
├── setup.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── vaepaster/
├── init.py
├── conditioned_align.py
├── unconditional_align.py
└── settings.yaml

## 🛠️ Installation

To install VaePaster, run the following command:

```bash
pip install vaepaster
```

## 🚦 Usage

VaePaster provides two main command-line interfaces:

1. Conditional Alignment:
   ```
   vaepaster-conditional [options]
   ```

2. Unconditional Alignment:
   ```
   vaepaster-unconditional [options]
   ```

For detailed usage instructions and available options, run the commands with the `--help` flag.

## ⚙️ Configuration

You can customize VaePaster's behavior by modifying the `settings.yaml` file. This file allows you to adjust various parameters for image processing and text recognition.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
