"""Accepted hand edits of the decoder, in order. Replay with
mas_sae.llm_edit.edit_rules.read_edits_py and apply_edits onto the original SAE."""

EDITS = [
    # edit 1, feature 4409:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 1, "feature": 4409, "row_file": "rows/edit_001.npy", "row_sha256": "a29058c9aa6ea080d811f58ac5f5b3dc441dff06f1b3949ebf5987ce37f372e6"},
    # edit 2, feature 12624:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 2, "feature": 12624, "row_file": "rows/edit_002.npy", "row_sha256": "4fe62be0935367cc32cd85a01306d2d7f7bc472267f1d90cbddbf21459ee835e"},
    # edit 3, feature 14120:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 3, "feature": 14120, "row_file": "rows/edit_003.npy", "row_sha256": "c19abfd33f0ddcf8509a2f3aa547f20c44adfb8dcd18f2d92ed921259e528e3f"},
    # edit 4, feature 10383:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 4, "feature": 10383, "row_file": "rows/edit_004.npy", "row_sha256": "7ae871cc88be507c8431fba1b4e9bd2f7a8800ae8557935757d0c4a23e572dff"},
    # edit 5, feature 12139:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 5, "feature": 12139, "row_file": "rows/edit_005.npy", "row_sha256": "fd379923940f3a4080e7a5f248c15d5aecbff609b23f544cecc24322929cd98c"},
    # edit 6, feature 1750:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 6, "feature": 1750, "row_file": "rows/edit_006.npy", "row_sha256": "424a50ade20945a5b96f925634a0091526d727d3aac9918e4716f74471747eb4"},
    # edit 7, feature 16138:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 7, "feature": 16138, "row_file": "rows/edit_007.npy", "row_sha256": "30250ab69008bc87aaf747d924e7cbc6e7af7808044dafe083642c534e63c262"},
    # edit 8, feature 16144:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 8, "feature": 16144, "row_file": "rows/edit_008.npy", "row_sha256": "2c6a10ff83dacb364437cd816f084dfd34226fd8449eca7ce34d0bb3524a03d1"},
    # edit 9, feature 7649:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 9, "feature": 7649, "row_file": "rows/edit_009.npy", "row_sha256": "d2b22c10be3827514c597a4da3da5191a9e7d4efa9cb303c53bf19fd6371e151"},
    # edit 10, feature 15005:
    #   mechanical: adoption row rotated toward d to cosine 0.9500099999999999
    {"edit": 10, "feature": 15005, "row_file": "rows/edit_010.npy", "row_sha256": "03eeaa34872d905ac02bb036f27d3a0a150a5853375d46340f1ef96b690ee737"},
    # edit 11, feature 11977:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 11, "feature": 11977, "row_file": "rows/edit_011.npy", "row_sha256": "e69afab7f765f31985f57bba5c187fcb70c44b2b919b925ac0490ea8a167c715"},
    # edit 12, feature 13643:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 12, "feature": 13643, "row_file": "rows/edit_012.npy", "row_sha256": "dda588f00c459eaff2ad571f350cda937c17d89b6b906efe4614c76e2b40a695"},
    # edit 13, feature 8278:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 13, "feature": 8278, "row_file": "rows/edit_013.npy", "row_sha256": "e5a10270f10378cdfa02f69fffbfac03d55a07e03a09fbe7beebaab729d13b17"},
    # edit 14, feature 11218:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 14, "feature": 11218, "row_file": "rows/edit_014.npy", "row_sha256": "def1843fca557afdb198f3e1318fd829862e207ae74c56903cade62364702336"},
    # edit 15, feature 125:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 15, "feature": 125, "row_file": "rows/edit_015.npy", "row_sha256": "7b7f9e1033ac7918985afade3601d39b68e8663e4cb46cc0dbbb04c1bb6f4370"},
    # edit 16, feature 10499:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 16, "feature": 10499, "row_file": "rows/edit_016.npy", "row_sha256": "7c2ba4c9ebc4797f6a40dfffaa95f91a790d008d288b90c9f9854b121d3efa12"},
    # edit 17, feature 657:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 17, "feature": 657, "row_file": "rows/edit_017.npy", "row_sha256": "722a3b09a7dc3f8ddacbfc0b679c9566112ee09044943dce49f0091089d75cf3"},
    # edit 18, feature 13887:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 18, "feature": 13887, "row_file": "rows/edit_018.npy", "row_sha256": "c96ff5aadbfcf530e82befac65f34c58a8e6d760eaa720942f1edf430602c55e"},
    # edit 19, feature 263:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 19, "feature": 263, "row_file": "rows/edit_019.npy", "row_sha256": "ace0defe252f409f1598b8539d6aa4370e03d74b6ef8970bc0f9cf13e13f6cb9"},
    # edit 20, feature 13626:
    #   mechanical: retention row rotated toward -d to cosine 0.9500099999999999
    {"edit": 20, "feature": 13626, "row_file": "rows/edit_020.npy", "row_sha256": "56a3287247eddf15785bdede5c9728a3fee74bf9b1c750543ea3657965612637"},
]
