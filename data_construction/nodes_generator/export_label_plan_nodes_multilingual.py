import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# =========================================================
# Language support
# =========================================================
SUPPORTED_LANGUAGES = {
    "zh": {
        "name": "Chinese",
        "supports_full": True,
        "accent_values": [
            "putonghua",
            "beijing_accent",
            "tianjin_accent",
            "sichuan_accent",
            "northeastern_accent",
            "taiwanese_mandarin",
            "cantonese_accented_mandarin",
            "shanghai_accented_mandarin",
        ],
    },
    "en": {
        "name": "English",
        "supports_full": True,
        "accent_values": [
            "american_accent",
            "british_accent",
            "indian_accent",
        ],
    },
    "ja": {"name": "Japanese", "supports_full": False, "accent_values": []},
    "ko": {"name": "Korean", "supports_full": False, "accent_values": []},
    "pt": {"name": "Portuguese", "supports_full": False, "accent_values": []},
    "fr": {"name": "French", "supports_full": False, "accent_values": []},
    "de": {"name": "German", "supports_full": False, "accent_values": []},
    "it": {"name": "Italian", "supports_full": False, "accent_values": []},
    "ru": {"name": "Russian", "supports_full": False, "accent_values": []},
    "es": {"name": "Spanish", "supports_full": False, "accent_values": []},
}

PRIMARY_LANGUAGES = {"zh", "en"}
MINI_RATIO = 0.30
MINI_MIN_ITEMS_PER_NODE = 4
FULL_LIMIT = 1000
MINI_LIMIT = 300


# =========================================================
# Core inventories
# =========================================================
CORE_VALUES = {
    # =========================
    # Timbre
    # =========================
    "age": [
        "child",
        "adult",
        "elderly",
    ],
    "gender": [
        "male",
        "female",
    ],
    "pitch": [
        "very_high",
        "high",
        "normal",
        "low",
        "very_low",
    ],
    "texture": [
        "deep",
        "rich",
        "breathy",
        "husky",
        "smoky",
        "magnetic",
        "soft",
        "bright",
        "clear",
        "nasal",
        "sweet",
        "mellow",
    ],

    # =========================
    # Style
    # =========================
    "speed": [
        "very_fast",
        "fast",
        "normal",
        "slow",
        "very_slow",
    ],
    "volume": [
        "very_quiet",
        "quiet",
        "normal",
        "loud",
        "very_loud",
    ],
    "emotion": [
        "neutral",
        "happy",
        "angry",
        "sad",
        "fearful",
        "surprised",
        "worried",
        "thoughtful",
    ],
    "personality": [
        "cheerful",
        "calm",
        "serious",
        "lively",
        "reserved",
        "humorous",
        "gentle",
        "confident",
        "aloof",
        "warm",
    ],
    "tone": [
        "firm",
        "euphemistic",
        "hesitant",
        "urging",
        "sarcastic",
        "complaining",
        "coquettish",
        "pleading",
        "indifferent",
        "playful",
        "awkward",
        "disappointed",
        "speechless",
        "disdainful",
        "teasing",
        "smug",
        "defiant",
    ],

    # =========================
    # Persona
    # =========================
    "scenario": [
        "news_broadcast",
        "poetry_recitation",
        "bedtime_story",
        "documentary_narration",
        "public_speech",
        "radio_hosting",
        "audiobook_narration",
        "weather_forecast",
        "ceremony_hosting",
        "sports_commentary",
    ],
    "social_role": [
        "teacher",
        "announcer",
        "street_vendor",
        "crosstalk_performer",
        "customer_service_agent",
        "tour_guide",
        "doctor",
        "auctioneer",
        "podcast_host",
        "live_stream_host",
    ],
    "stylized_persona": [
        "yujie",
        "loli",
        "mischievous_child",
        "tomboyish_youth",
        "shy_girl",
        "gentle_sister",
        "tsundere_girl",
        "sunny_boy",
        "aloof_young_man",
        "spoiled_brat",
    ],
    "non_ip_archetype": [
        "princess",
        "queen",
        "undead_spirit",
        "priestess",
        "wizard",
        "witch",
        "knight",
        "elf",
        "vampire_noble",
        "ghost",
    ],
    "ip_portrait": [
        "taiyi_zhenren_nezha",
        "woqishanyi_demon_slayer",
        "joker_batman",
        "dingzhen",
        "spongebob",
        "klee_genshin",
        "zengxiaoxian",
        "xiaoyu_jackie_chan",
        "sunwukong_journey",
        "crayon_shinchan",
        "doraemon",
        "harry_potter",
        "elsa_frozen",
        "deadpool",
        "po_kungfu_panda",
        "maleficent",
    ],

    # =========================
    # Extra-vocal
    # =========================
    "disfluency": [
        "stutter",
        "repetition",
        "blocking",
        "hesitant_onset",
    ],
    "dysphonia": [
        "breathless",
        "hoarse",
        "trembling_voice",
        "weak_support",
        "strained_voice",
        "post_run_state",
        "post_cry_state",
        "fatigued_voice",
    ],
    "implicit_nv": [
        "breath",
        "inhale",
        "exhale",
        "quick_breath",
        "sigh",
        "gasp",
        "panting",
        "wheezing",
        "snore",
        "yawn",
        "cough",
        "sneeze",
        "throat_clearing",
        "hiccup",
        "sniff",
        "sniffle",
        "snort",
        "chuckle",
        "giggle",
        "laugh",
        "laugh_harder",
        "start_laughing",
        "stifled_laugh",
        "burst_of_laughter",
        "crying",
        "sobbing",
        "crying_loudly",
        "wail",
        "whimper",
        "hum",
        "humming",
        "groan",
        "moan",
        "grunt",
        "mumble",
        "exclamation",
        "lip_smack",
        "gulp",
        "swallow",
        "burp",
        "tsk",
        "sss",
        "clucking",
        "hissing",
        "whisper",
    ],
    "explicit_nv": [
        "breath",
        "inhale",
        "exhale",
        "quick_breath",
        "sigh",
        "gasp",
        "panting",
        "wheezing",
        "snore",
        "yawn",
        "cough",
        "sneeze",
        "throat_clearing",
        "hiccup",
        "sniff",
        "sniffle",
        "snort",
        "chuckle",
        "giggle",
        "laugh",
        "laugh_harder",
        "start_laughing",
        "stifled_laugh",
        "burst_of_laughter",
        "crying",
        "sobbing",
        "crying_loudly",
        "wail",
        "whimper",
        "hum",
        "humming",
        "groan",
        "moan",
        "grunt",
        "mumble",
        "exclamation",
        "lip_smack",
        "gulp",
        "swallow",
        "burp",
        "tsk",
        "sss",
        "clucking",
        "hissing",
        "whisper",
    ],
}


IP_PORTRAIT_ANCHORS = {
    "taiyi_zhenren_nezha": {
        "franchise_or_title": "哪吒之魔童降世",
        "character_name": "太乙真人",
        "portrait_hint": "middle-aged male, slightly tipsy, Sichuan-accented Mandarin, relaxed and humorous, sloppy on the outside but warm-hearted underneath",
    },
    "woqishanyi_demon_slayer": {
        "franchise_or_title": "鬼灭之刃",
        "character_name": "我妻善逸",
        "portrait_hint": "young male, relatively high-pitched, nervous and excitable, emotionally volatile, sharp contrast between panic and intensity",
    },
    "joker_batman": {
        "franchise_or_title": "蝙蝠侠",
        "character_name": "小丑",
        "portrait_hint": "male, theatrical, unstable, eerie laughter, provocative rhythm, dangerous playfulness, emotionally unpredictable",
    },
    "dingzhen": {
        "franchise_or_title": "现实人物语境",
        "character_name": "丁真",
        "portrait_hint": "young ethnic-minority male impression, relatively clean and natural voice, simple and unpolished delivery, faint regional coloring, innocent and straightforward temperament",
    },
    "spongebob": {
        "franchise_or_title": "海绵宝宝",
        "character_name": "海绵宝宝",
        "portrait_hint": "cartoonish high-pitched voice, bright and springy rhythm, exaggerated innocence, energetic and slightly silly expressiveness",
    },
    "klee_genshin": {
        "franchise_or_title": "原神",
        "character_name": "可莉",
        "portrait_hint": "very young girl impression, high pitch, bouncy and lively rhythm, innocent mischievousness, playful explosive energy",
    },
    "zengxiaoxian": {
        "franchise_or_title": "爱情公寓",
        "character_name": "曾小贤",
        "portrait_hint": "adult male, slightly oily and comedic talk-show style, self-conscious performative rhythm, exaggerated but not fully confident",
    },
    "xiaoyu_jackie_chan": {
        "franchise_or_title": "成龙历险记",
        "character_name": "小玉",
        "portrait_hint": "young girl impression, lively and clever, quick and mischievous delivery, energetic and rebellious undertone",
    },
    "sunwukong_journey": {
        "franchise_or_title": "西游记相关文化原型",
        "character_name": "孙悟空",
        "portrait_hint": "nimble masculine voice, bright and agile projection, playful rebelliousness, confident heroic energy, quick rhythmic attack",
    },
    "crayon_shinchan": {
        "franchise_or_title": "蜡笔小新",
        "character_name": "野原新之助",
        "portrait_hint": "boyish voice with mischievous laziness, nasal and draggy rhythm, cheeky irreverence, comic childlike shamelessness",
    },
    "doraemon": {
        "franchise_or_title": "哆啦A梦",
        "character_name": "哆啦A梦",
        "portrait_hint": "round cartoon voice, warm but slightly clumsy authority, soft and friendly timbre, comforting yet comedic delivery",
    },
    "harry_potter": {
        "franchise_or_title": "哈利波特",
        "character_name": "哈利·波特",
        "portrait_hint": "teenage boy impression, clear youthful tone, earnest and restrained bravery, polite but determined delivery",
    },
    "elsa_frozen": {
        "franchise_or_title": "冰雪奇缘",
        "character_name": "艾莎",
        "portrait_hint": "adult female, cool and elegant tone, controlled breath support, noble restraint, emotionally distant but powerful inner tension",
    },
    "deadpool": {
        "franchise_or_title": "死侍",
        "character_name": "Deadpool",
        "portrait_hint": "adult male, glib and fast-talking, sarcastic and irreverent, playful aggression, unstable seriousness with comic timing",
    },
    "po_kungfu_panda": {
        "franchise_or_title": "功夫熊猫",
        "character_name": "阿宝",
        "portrait_hint": "warm and slightly heavy cartoon male voice, earnest enthusiasm, comic clumsiness, food-loving friendliness, optimistic drive",
    },
    "maleficent": {
        "franchise_or_title": "沉睡魔咒",
        "character_name": "黑魔女",
        "portrait_hint": "adult female, dark and imposing elegance, low and controlled energy, aristocratic coldness, dangerous calmness with pressure",
    },
}


MINT_BENCH_NODE_SPECS = [
    {"taxonomy_node": "easy/timbre/tag/atomic/age", "num_items": 12, "allowed_labels": ["age"]},
    {"taxonomy_node": "easy/timbre/tag/atomic/gender", "num_items": 10, "allowed_labels": ["gender"]},
    {"taxonomy_node": "easy/timbre/tag/atomic/pitch", "num_items": 15, "allowed_labels": ["pitch"]},
    {"taxonomy_node": "easy/timbre/tag/atomic/texture", "num_items": 24, "allowed_labels": ["texture"]},
    {"taxonomy_node": "easy/timbre/direct/atomic/age", "num_items": 12, "allowed_labels": ["age"]},
    {"taxonomy_node": "easy/timbre/direct/atomic/gender", "num_items": 10, "allowed_labels": ["gender"]},
    {"taxonomy_node": "easy/timbre/direct/atomic/pitch", "num_items": 15, "allowed_labels": ["pitch"]},
    {"taxonomy_node": "easy/timbre/direct/atomic/texture", "num_items": 24, "allowed_labels": ["texture"]},
    {"taxonomy_node": "easy/style/tag/atomic/speed", "num_items": 15, "allowed_labels": ["speed"]},
    {"taxonomy_node": "easy/style/tag/atomic/volume", "num_items": 15, "allowed_labels": ["volume"]},
    {"taxonomy_node": "easy/style/tag/atomic/emotion", "num_items": 24, "allowed_labels": ["emotion"]},
    {
        "taxonomy_node": "easy/style/tag/atomic/accent",
        "num_items": 24,
        "allowed_labels": ["accent"],
        "requires_accent": True,
    },
    {"taxonomy_node": "easy/style/tag/atomic/personality", "num_items": 20, "allowed_labels": ["personality"]},
    {"taxonomy_node": "easy/style/tag/atomic/tone", "num_items": 34, "allowed_labels": ["tone"]},
    {"taxonomy_node": "easy/style/direct/atomic/speed", "num_items": 15, "allowed_labels": ["speed"]},
    {"taxonomy_node": "easy/style/direct/atomic/volume", "num_items": 15, "allowed_labels": ["volume"]},
    {"taxonomy_node": "easy/style/direct/atomic/emotion", "num_items": 24, "allowed_labels": ["emotion"]},
    {
        "taxonomy_node": "easy/style/direct/atomic/accent",
        "num_items": 24,
        "allowed_labels": ["accent"],
        "requires_accent": True,
    },
    {"taxonomy_node": "easy/style/direct/atomic/personality", "num_items": 20, "allowed_labels": ["personality"]},
    {"taxonomy_node": "easy/style/direct/atomic/tone", "num_items": 34, "allowed_labels": ["tone"]},
    {
        "taxonomy_node": "easy/composite/simple_comp/static",
        "num_items": 72,
        "allowed_labels": ["age", "gender", "pitch", "texture", "speed", "volume", "emotion", "accent", "personality", "tone"],
    },
    {
        "taxonomy_node": "hard/composite/complex_comp/static",
        "num_items": 72,
        "allowed_labels": ["age", "gender", "pitch", "texture", "speed", "volume", "emotion", "accent", "personality", "tone"],
    },
    {
        "taxonomy_node": "hard/composite/complex_comp/dynamic",
        "num_items": 36,
        "allowed_labels": ["pitch", "speed", "volume", "emotion", "tone"],
    },
    {
        "taxonomy_node": "hard/composite/complex_comp/layered",
        "num_items": 32,
        "allowed_labels": ["emotion", "personality", "tone"],
    },
    {
        "taxonomy_node": "hard/composite/complex_comp/conflict",
        "num_items": 32,
        "allowed_labels": ["age", "gender", "pitch", "texture", "speed", "volume", "emotion", "accent", "personality", "tone"],
    },
    {"taxonomy_node": "hard/composite/persona/scenario", "num_items": 30, "allowed_labels": ["scenario"]},
    {"taxonomy_node": "hard/composite/persona/social_role", "num_items": 30, "allowed_labels": ["social_role"]},
    {"taxonomy_node": "hard/composite/persona/stylized_persona", "num_items": 30, "allowed_labels": ["stylized_persona"]},
    {"taxonomy_node": "hard/composite/persona/non_ip_archetype", "num_items": 30, "allowed_labels": ["non_ip_archetype"]},
    {
        "taxonomy_node": "hard/composite/persona/ip_portrait",
        "num_items": 32,
        "allowed_labels": ["ip_portrait"],
        "aux_value_lookup_key": "ip_portrait",
        "aux_value_lookup": IP_PORTRAIT_ANCHORS,
    },
    {"taxonomy_node": "special/extra_vocal/abnormal/disfluency", "num_items": 20, "allowed_labels": ["disfluency"]},
    {"taxonomy_node": "special/extra_vocal/abnormal/dysphonia", "num_items": 24, "allowed_labels": ["dysphonia"]},
    {"taxonomy_node": "special/extra_vocal/implicit_nv", "num_items": 32, "allowed_labels": ["implicit_nv"]},
    {"taxonomy_node": "special/extra_vocal/explicit_nv", "num_items": 32, "allowed_labels": ["explicit_nv"]},
]


def get_accent_values(language: str):
    return SUPPORTED_LANGUAGES[language]["accent_values"]


def supports_accent(language: str) -> bool:
    return bool(get_accent_values(language))


def normalize_preset(preset: str) -> str:
    normalized = str(preset).strip().lower()
    if normalized == "smoke":
        return "mini"
    if normalized not in {"full", "mini"}:
        raise ValueError("Preset must be one of: full, mini, smoke")
    return normalized


def normalize_languages(languages: Optional[Sequence[str]]) -> List[str]:
    if not languages:
        return []

    normalized = []
    for value in languages:
        if value is None:
            continue
        for chunk in str(value).split(","):
            language = chunk.strip().lower()
            if not language:
                continue
            if language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {language}")
            if language not in normalized:
                normalized.append(language)
    return normalized


def normalize_taxonomy_filters(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []

    normalized = []
    for value in values:
        if value is None:
            continue
        for chunk in str(value).split(","):
            item = chunk.strip().strip("/")
            if not item:
                continue
            if item not in normalized:
                normalized.append(item)
    return normalized


def filter_node_specs(
    specs: Sequence[dict],
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
) -> List[dict]:
    allowed_nodes = set(normalize_taxonomy_filters(taxonomy_nodes))
    allowed_prefixes = normalize_taxonomy_filters(taxonomy_prefixes)

    if not allowed_nodes and not allowed_prefixes:
        return list(specs)

    filtered = []
    for spec in specs:
        taxonomy_node = str(spec["taxonomy_node"]).strip("/")
        if allowed_nodes and taxonomy_node in allowed_nodes:
            filtered.append(spec)
            continue
        if allowed_prefixes and any(taxonomy_node.startswith(prefix) for prefix in allowed_prefixes):
            filtered.append(spec)
    return filtered


def pick_values(keys, language: str):
    values = {}
    for key in keys:
        if key == "accent":
            accent_values = get_accent_values(language)
            if accent_values:
                values[key] = accent_values
            continue
        values[key] = CORE_VALUES[key]
    return values


def normalize_labels_for_language(labels, language: str):
    if supports_accent(language):
        return list(labels)
    return [label for label in labels if label != "accent"]


def derive_num_items(full_num_items: int, preset: str) -> int:
    normalized_preset = normalize_preset(preset)
    if normalized_preset == "full":
        return full_num_items
    if normalized_preset == "mini":
        return max(MINI_MIN_ITEMS_PER_NODE, round(full_num_items * MINI_RATIO))
    raise ValueError("Preset must be one of: full, mini, smoke")


# =========================================================
# Builders
# =========================================================
def build_node(spec: dict, language: str, preset: str):
    if spec.get("requires_accent") and not supports_accent(language):
        return None

    allowed_labels = normalize_labels_for_language(spec["allowed_labels"], language)
    if not allowed_labels:
        return None

    node = {
        "taxonomy_node": spec["taxonomy_node"],
        "num_items": derive_num_items(spec["num_items"], preset),
        "allowed_labels": allowed_labels,
        "allowed_values": pick_values(allowed_labels, language),
        "language": language,
    }

    if "aux_value_lookup_key" in spec:
        node["aux_value_lookup_key"] = spec["aux_value_lookup_key"]
    if "aux_value_lookup" in spec:
        node["aux_value_lookup"] = spec["aux_value_lookup"]

    return node



def build_label_plan_nodes(
    preset: str,
    language: str,
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
):
    normalized_preset = normalize_preset(preset)
    language = language.lower()

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    if normalized_preset == "full" and not SUPPORTED_LANGUAGES[language]["supports_full"]:
        raise ValueError(
            f"language={language} only supports the mini preset; full is available only for zh and en."
        )

    filtered_specs = filter_node_specs(
        MINT_BENCH_NODE_SPECS,
        taxonomy_nodes=taxonomy_nodes,
        taxonomy_prefixes=taxonomy_prefixes,
    )

    nodes = []
    for spec in filtered_specs:
        node = build_node(spec, language=language, preset=normalized_preset)
        if node is not None:
            nodes.append(node)

    return nodes


# Backward-compatible defaults for existing imports.
LABEL_PLAN_NODES_FULL = build_label_plan_nodes(preset="full", language="zh")
LABEL_PLAN_NODES_MINI = build_label_plan_nodes(preset="mini", language="zh")
LABEL_PLAN_NODES_SMOKE = LABEL_PLAN_NODES_MINI
TOTAL_NUM_ITEMS_FULL = sum(x["num_items"] for x in LABEL_PLAN_NODES_FULL)
TOTAL_NUM_ITEMS_MINI = sum(x["num_items"] for x in LABEL_PLAN_NODES_MINI)
TOTAL_NUM_ITEMS_SMOKE = TOTAL_NUM_ITEMS_MINI


# =========================================================
# Export helpers
# =========================================================
def default_output_filename(preset: str, language: str) -> str:
    normalized_preset = normalize_preset(preset)
    return f"label_plan_nodes_{normalized_preset}_{language}.json"



def export_label_plan_config(
    preset: str,
    output_path: str,
    language: str = "zh",
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
):
    normalized_preset = normalize_preset(preset)
    nodes = build_label_plan_nodes(
        preset=normalized_preset,
        language=language,
        taxonomy_nodes=taxonomy_nodes,
        taxonomy_prefixes=taxonomy_prefixes,
    )
    total = sum(x["num_items"] for x in nodes)
    limit = FULL_LIMIT if normalized_preset == "full" else MINI_LIMIT

    if total > limit:
        raise ValueError(
            f"Exported {normalized_preset} config exceeds the limit: total={total}, limit={limit}, language={language}"
        )

    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    print(f"[OK] Exported {normalized_preset} label-plan config to: {output_path}")
    print(f"[INFO] Language: {language} ({SUPPORTED_LANGUAGES[language]['name']})")
    print(f"[INFO] Node count: {len(nodes)}")
    print(f"[INFO] Total num_items: {total}")
    print(f"[INFO] Accent enabled: {supports_accent(language)}")



def export_label_plan_configs(
    preset: str,
    languages: Sequence[str],
    output_dir: str,
    taxonomy_nodes: Optional[Sequence[str]] = None,
    taxonomy_prefixes: Optional[Sequence[str]] = None,
):
    normalized_preset = normalize_preset(preset)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for language in languages:
        output_path = output_dir_path / default_output_filename(normalized_preset, language)
        export_label_plan_config(
            preset=normalized_preset,
            output_path=str(output_path),
            language=language,
            taxonomy_nodes=taxonomy_nodes,
            taxonomy_prefixes=taxonomy_prefixes,
        )



def list_supported_taxonomy_nodes() -> List[str]:
    return [spec["taxonomy_node"] for spec in MINT_BENCH_NODE_SPECS]



def print_supported_values(title: str, values: Iterable[str]) -> None:
    print(title)
    for value in values:
        print(f"- {value}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export multilingual node specifications for MINT-Bench data construction. "
            "This stage defines the benchmark specification layer, including attributes, values, "
            "taxonomy coverage, and per-node sample counts."
        )
    )
    parser.add_argument(
        "--preset",
        "--mode",
        dest="preset",
        type=str,
        choices=["full", "mini", "smoke"],
        default="mini",
        help="Benchmark preset to export. 'mini' is the public name; 'smoke' is accepted as a compatibility alias.",
    )
    parser.add_argument(
        "--language",
        action="append",
        help="Language to export. Repeat the flag or provide a comma-separated list.",
    )
    parser.add_argument(
        "--taxonomy-node",
        action="append",
        help="Exact taxonomy node to include. Repeat the flag or provide a comma-separated list.",
    )
    parser.add_argument(
        "--taxonomy-prefix",
        action="append",
        help="Taxonomy prefix to include, such as easy/style or hard/composite. Repeat the flag or provide a comma-separated list.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for a single exported config file. Use this when exporting one language.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for one or more exported config files. Files are named automatically.",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported languages and exit.",
    )
    parser.add_argument(
        "--list-taxonomy-nodes",
        action="store_true",
        help="List all supported taxonomy nodes and exit.",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    if args.list_languages:
        print_supported_values("Supported languages:", sorted(SUPPORTED_LANGUAGES.keys()))
        return

    if args.list_taxonomy_nodes:
        print_supported_values("Supported taxonomy nodes:", list_supported_taxonomy_nodes())
        return

    languages = normalize_languages(args.language) or ["zh"]
    taxonomy_nodes = normalize_taxonomy_filters(args.taxonomy_node)
    taxonomy_prefixes = normalize_taxonomy_filters(args.taxonomy_prefix)
    normalized_preset = normalize_preset(args.preset)

    if args.output_path and args.output_dir:
        raise ValueError("Use either --output_path or --output_dir, not both.")

    if args.output_path:
        if len(languages) != 1:
            raise ValueError("--output_path requires exactly one language. Use --output_dir for multi-language export.")
        export_label_plan_config(
            preset=normalized_preset,
            output_path=args.output_path,
            language=languages[0],
            taxonomy_nodes=taxonomy_nodes,
            taxonomy_prefixes=taxonomy_prefixes,
        )
        return

    output_dir = args.output_dir
    if not output_dir:
        script_dir = Path(__file__).resolve().parent
        output_dir = str(script_dir / "nodes")

    export_label_plan_configs(
        preset=normalized_preset,
        languages=languages,
        output_dir=output_dir,
        taxonomy_nodes=taxonomy_nodes,
        taxonomy_prefixes=taxonomy_prefixes,
    )


if __name__ == "__main__":
    main()
