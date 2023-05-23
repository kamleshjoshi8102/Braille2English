function convertToBraille(text) {
    const brailleMap = {
      "a": "⠁",
      "b": "⠃",
      "c": "⠉",
      "d": "⠙",
      "e": "⠑",
      "f": "⠋",
      "g": "⠛",
      "h": "⠓",
      "i": "⠊",
      "j": "⠚",
      "k": "⠅",
      "l": "⠇",
      "m": "⠍",
      "n": "⠝",
      "o": "⠕",
      "p": "⠏",
      "q": "⠟",
      "r": "⠗",
      "s": "⠎",
      "t": "⠞",
      "u": "⠥",
      "v": "⠧",
      "w": "⠺",
      "x": "⠭",
      "y": "⠽",
      "z": "⠵",
      " ": " ",
    };
  
    let brailleText = "";
    for (let i = 0; i < text.length; i++) {
      const char = text.charAt(i).toLowerCase();
      if (brailleMap[char]) {
        brailleText += brailleMap[char];
      } else {
        brailleText += char;
      }
    }
    return brailleText;
  }
  