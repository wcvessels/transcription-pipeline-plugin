import transcribe

A1_0_PRESENT = {
    "--model", "--language", "--keep-audio", "--output-dir", "--scene-threshold",
    "--max-frames", "--interval-seconds", "--frames-per-minute", "--window-size", "--dedup-threshold",
    "--allow-low-quality-frames", "--force-transcribe", "--prefer-captions", "--keep-work",
    "--diarize", "--source-hint",
}
MUST_BE_ABSENT = {
    "--no-diarize", "--ai-assist", "--curate", "--align", "--ocr-engine", "--polish",
    "--compose-with-claude", "--format", "--api-model", "--resume", "--force-local",
}


def test_flag_set_matches_section_16_3_exactly():
    p = transcribe.build_parser()
    names = {a.option_strings[0] for a in p._actions if a.option_strings and a.option_strings[0] != "-h"}
    assert A1_0_PRESENT <= names, f"missing flags: {A1_0_PRESENT - names}"
    assert not (MUST_BE_ABSENT & names), f"forbidden flags present: {MUST_BE_ABSENT & names}"
    # no unexpected extras beyond the locked set (and the implicit -h/--help)
    extras = names - A1_0_PRESENT - {"--help"}
    assert not extras, f"undocumented flags in parser: {extras}"
