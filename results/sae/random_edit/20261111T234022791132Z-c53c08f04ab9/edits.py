"""Accepted hand edits of the decoder, in order. Replay with
mas_sae.llm_edit.edit_rules.read_edits_py and apply_edits onto the original SAE."""

EDITS = [
    # edit 1, feature 4409:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 1, "feature": 4409, "row_file": "rows/edit_001.npy", "row_sha256": "b15d4f9c1ef30baf342bd9e8e20049466c627507afbc18340a18811294e107c0"},
    # edit 2, feature 12624:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 2, "feature": 12624, "row_file": "rows/edit_002.npy", "row_sha256": "c0989d9e4c3850b4c281df0aa260205cf39c01424dba3dec6a3f8920a4f20de4"},
    # edit 3, feature 14120:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 3, "feature": 14120, "row_file": "rows/edit_003.npy", "row_sha256": "83e7661e30bf6af8d6a4ae96122a7104d80a59a25511320d47e7403a0ba317b6"},
    # edit 4, feature 10383:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 4, "feature": 10383, "row_file": "rows/edit_004.npy", "row_sha256": "03740436bdc323a5893ae5a4e7c91ad544b05920b3993911a6ee19ee1e06f6f8"},
    # edit 5, feature 12139:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 5, "feature": 12139, "row_file": "rows/edit_005.npy", "row_sha256": "5ccc93971dff92dbdef085901acbd2470eb5f383707cb474820b7576f66b3a1e"},
    # edit 6, feature 1750:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 6, "feature": 1750, "row_file": "rows/edit_006.npy", "row_sha256": "4877f96f6a9b3cf4ad1261ded7a6f8bf5977028c40787f48f0addb5edd63fdcd"},
    # edit 7, feature 16138:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 7, "feature": 16138, "row_file": "rows/edit_007.npy", "row_sha256": "e91dc1d6f9e658302aa558741957fec13757aee4d1e2660695d00137194b09aa"},
    # edit 8, feature 16144:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 8, "feature": 16144, "row_file": "rows/edit_008.npy", "row_sha256": "0d64715b792c5da0766c2a2097f451c55aec29eedb4558dabc026fe6e0a68e43"},
    # edit 9, feature 7649:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 9, "feature": 7649, "row_file": "rows/edit_009.npy", "row_sha256": "3450f35d619f274b62702322a5cd368e2012deec1c59971fe3eb9fba25ecbaa6"},
    # edit 10, feature 15005:
    #   random: adoption row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 10, "feature": 15005, "row_file": "rows/edit_010.npy", "row_sha256": "93eaced490fb2e663cb302dacc36895dca76bd02d8285a277aa9143495be4b4a"},
    # edit 11, feature 11977:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 11, "feature": 11977, "row_file": "rows/edit_011.npy", "row_sha256": "76c8692d7f85c9cc530a61c7990e19a26b49b4315c652bbc8df137807c9e6e3c"},
    # edit 12, feature 13643:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 12, "feature": 13643, "row_file": "rows/edit_012.npy", "row_sha256": "a18e724b5806b9cb138c1fcb0bdc16739f33cedd6f1ee7acafc49485791d43be"},
    # edit 13, feature 8278:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 13, "feature": 8278, "row_file": "rows/edit_013.npy", "row_sha256": "16545fc04d84869573e7deebbc36686e010ea1f4b900c654932034f4be570969"},
    # edit 14, feature 11218:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 14, "feature": 11218, "row_file": "rows/edit_014.npy", "row_sha256": "81fac85408f5992aa96c58e0ecbc2d535e7e801eff56ce7e524a4d8f0d7c5b94"},
    # edit 15, feature 125:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 15, "feature": 125, "row_file": "rows/edit_015.npy", "row_sha256": "b5eb96461bffdbd7040f76ec4af15e985c2d722ffd86444394cf4a595ef1e350"},
    # edit 16, feature 10499:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 16, "feature": 10499, "row_file": "rows/edit_016.npy", "row_sha256": "23d2f137893a3aaaf551274b28e32c24be24ce711857589dfd027b30a860c55d"},
    # edit 17, feature 657:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 17, "feature": 657, "row_file": "rows/edit_017.npy", "row_sha256": "de4fe6f89359ed70170c58c2e274d4c1b67ca1a8beb425be47a4a4718f65d146"},
    # edit 18, feature 13887:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 18, "feature": 13887, "row_file": "rows/edit_018.npy", "row_sha256": "61905e80c3ab097b99aca5f8a7676652f8936045397e8f331add59308b7f1a67"},
    # edit 19, feature 263:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 19, "feature": 263, "row_file": "rows/edit_019.npy", "row_sha256": "a655f1de61bc718bab2e69d8529187d368fae4e139d88d848796164bf055f055"},
    # edit 20, feature 13626:
    #   random: retention row rotated toward a random direction to the LLM edit's cosine 0.955000
    {"edit": 20, "feature": 13626, "row_file": "rows/edit_020.npy", "row_sha256": "87cb3c8df8496a01231fc536aae4ac9b07e2347f2dd6c3ced0b7a2210c75a641"},
]
