from pathlib import Path


def main():
    path = Path(r"C:/Users/ryoi/AppData/Local/anaconda3/envs/newphy2/Lib/site-packages/phylib/io/model.py")
    lines = path.read_text().splitlines()
    cleaned = []
    blank_streak = 0
    for line in lines:
        if line.strip() == "":
            blank_streak += 1
            if blank_streak > 1:
                continue
        else:
            blank_streak = 0
        cleaned.append(line)
    path.write_text("\r\n".join(cleaned) + "\r\n")
    print(f"Collapsed blank lines; new line count: {len(cleaned)}")


if __name__ == "__main__":
    main()
