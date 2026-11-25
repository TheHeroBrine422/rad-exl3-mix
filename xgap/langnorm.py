"""langnorm.py -- one language scheme for xgap-v1 manifests: ISO 639-1 where one exists (else the ISO 639-3 code), plus a
script suffix only for languages with several living scripts (sr-Cyrl, kk-Cyrl, ...). Programming languages (code cells)
keep their lowercase name. The source-native value is kept as `lang_src`."""
import re

# ISO 639-3 -> 639-1 (the languages of FineWeb-2 / FineWeb2-HQ / Aya / the Wikipedias used here)
I3 = dict(afr="af", als="sq", sqi="sq", amh="am", arb="ar", ara="ar", arz="ar", azj="az", aze="az", eus="eu", bel="be", ben="bn", bul="bg", mya="my",
          cat="ca", zho="zh", cmn="zh", yue="zh", hrv="hr", ces="cs", dan="da", nld="nl", eng="en", est="et", fin="fi", fra="fr", glg="gl", kat="ka",
          deu="de", ell="el", guj="gu", hat="ht", hau="ha", heb="he", hin="hi", hun="hu", isl="is", ibo="ig", ind="id", gle="ga", ita="it", jpn="ja",
          kan="kn", kaz="kk", khm="km", kor="ko", kir="ky", lao="lo", lav="lv", lit="lt", mkd="mk", zsm="ms", msa="ms", mal="ml", mlt="mt", mar="mr",
          npi="ne", nep="ne", nor="no", nob="nb", nya="ny", pan="pa", pes="fa", fas="fa", plt="mg", pol="pl", por="pt", ron="ro", rus="ru", srp="sr",
          sin="si", slk="sk", slv="sl", som="so", pbt="ps", pus="ps", spa="es", sun="su", swh="sw", swa="sw", swe="sv", tam="ta", tel="te", tha="th",
          tur="tr", ukr="uk", urd="ur", vie="vi", cym="cy", wol="wo", xho="xh", yor="yo", zul="zu", fil="fil", tgl="tl", uzn="uz", uzb="uz", tgk="tg",
          mon="mn", khk="mn", hye="hy", bos="bs", kur="ku", ckb="ku", lat="la", epo="eo", sna="sn", sot="st", tir="ti", ory="or", asm="as", snd="sd")
# languages that are normally written in more than one script: keep the script when the source gives it
MULTI_SCRIPT = {"sr", "kk", "uz", "az", "mn", "pa", "ku", "tg", "bs", "ms"}
NAMES = {   # WildChat's language names
    "english": "en", "chinese": "zh", "russian": "ru", "french": "fr", "spanish": "es", "portuguese": "pt", "german": "de", "italian": "it",
    "japanese": "ja", "korean": "ko", "arabic": "ar", "turkish": "tr", "vietnamese": "vi", "polish": "pl", "dutch": "nl", "indonesian": "id",
    "thai": "th", "hindi": "hi", "hebrew": "he", "ukrainian": "uk", "persian": "fa", "swedish": "sv", "czech": "cs", "greek": "el",
    "hungarian": "hu", "romanian": "ro", "finnish": "fi", "danish": "da", "norwegian": "no", "bulgarian": "bg", "slovak": "sk", "croatian": "hr",
    "serbian": "sr", "catalan": "ca", "lithuanian": "lt", "latvian": "lv", "estonian": "et", "slovenian": "sl", "bengali": "bn", "tamil": "ta",
    "urdu": "ur", "malay": "ms", "filipino": "fil", "tagalog": "tl", "swahili": "sw", "afrikaans": "af", "icelandic": "is", "macedonian": "mk",
    "albanian": "sq", "armenian": "hy", "azerbaijani": "az", "basque": "eu", "belarusian": "be", "georgian": "ka", "kazakh": "kk", "latin": "la",
    "mongolian": "mn", "nepali": "ne", "telugu": "te", "marathi": "mr", "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa",
    "sinhala": "si", "khmer": "km", "lao": "lo", "burmese": "my", "uzbek": "uz", "welsh": "cy", "irish": "ga", "galician": "gl", "maltese": "mt",
    "esperanto": "eo", "yoruba": "yo", "zulu": "zu", "xhosa": "xh", "somali": "so", "hausa": "ha", "amharic": "am", "pashto": "ps", "kurdish": "ku"}


def norm(adapter, gen, raw):
    """-> normalized language tag. adapter / gen come from the source spec."""
    if raw is None: return None
    raw = str(raw)
    if adapter in ("stack", "stackedu"):
        return raw.lower().replace("c++", "cpp").replace("c#", "csharp")
    if adapter == "wildchat":
        return NAMES.get(raw.strip().lower(), "und")
    m = re.fullmatch(r"([a-z]{3})_([A-Z][a-z]{3})", raw)                # FineWeb-2: kor_Hang
    if m:
        l = I3.get(m.group(1), m.group(1))
        return f"{l}-{m.group(2)}" if l in MULTI_SCRIPT else l
    base = re.split(r"[-_]", raw)[0].lower()
    if len(base) == 3: return I3.get(base, base)                          # Aya: arb, fra, plt, ...
    return base                                                            # wikipedia / oasst2 / 'en': two letters
