lang_scripts=(
    eng_Latn
    hun_Latn
    vie_Latn
    fas_Arab
    hin_Deva
    srp_Cyrl
    gla_Latn
    mri_Latn
    tpi_Latn
    sot_Latn
    amh_Ethi
    guj_Gujr
    ory_Orya
    mya_Mymr
)

for lang_script in "${lang_scripts[@]}"; do
    python download_glot500c.py --lang_script "$lang_script"
done
