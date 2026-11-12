"""Accepted hand edits of the decoder, in order. Replay with
mas_sae.llm_edit.edit_rules.read_edits_py and apply_edits onto the original SAE."""

EDITS = [
    # edit 1, feature 4409:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 1, "feature": 4409, "row_file": "rows/edit_001.npy", "row_sha256": "20a24b789fdc4b0e8bee340d73c38a272f9c85df62105a64d27f0dec6464a119"},
    # edit 2, feature 12624:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 2, "feature": 12624, "row_file": "rows/edit_002.npy", "row_sha256": "7cfc12e139f459a6e8ce3de4b0dabf6c99dd9c532cc87c555a5785f2ab1096ab"},
    # edit 3, feature 14120:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 3, "feature": 14120, "row_file": "rows/edit_003.npy", "row_sha256": "c1e73b54450b7daddcf34f09e69add6485d1fcba93c0a6512e775d97a8ccdae2"},
    # edit 4, feature 10383:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 4, "feature": 10383, "row_file": "rows/edit_004.npy", "row_sha256": "0efeaa1f010dbf9ba95c093ad63ce32c877e246fa98dac91db3eb1a40cd7ab57"},
    # edit 5, feature 12139:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 5, "feature": 12139, "row_file": "rows/edit_005.npy", "row_sha256": "a95448bb91da5a57a7775f44c7e000222f4e1d2bbd8a2534eb6c60b28575d642"},
    # edit 6, feature 1750:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 6, "feature": 1750, "row_file": "rows/edit_006.npy", "row_sha256": "2139377eab022463d57cc1945f17f7bc49049444505eca112a2debf7bef89428"},
    # edit 7, feature 16138:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 7, "feature": 16138, "row_file": "rows/edit_007.npy", "row_sha256": "7306ee1aad0174199afd4ad1f68a93db183e07a1170e1cff972927af35b875df"},
    # edit 8, feature 16144:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 8, "feature": 16144, "row_file": "rows/edit_008.npy", "row_sha256": "3b73199f4814e94cbe6cefef41273359049e3b1918d86978b9c9f34b2dfb65be"},
    # edit 9, feature 7649:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 9, "feature": 7649, "row_file": "rows/edit_009.npy", "row_sha256": "35b21789cd2d7a37ce7a9969690b8954d0db54be89702327529329093a6f9b12"},
    # edit 10, feature 15005:
    #   adoption row rotated 0.32 of a unit step toward +d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 10, "feature": 15005, "row_file": "rows/edit_010.npy", "row_sha256": "b5d1b1a630becfbc38bb3412c56993b13dd3797013613f639e17b4a9e3f497a2"},
    # edit 11, feature 11977:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 11, "feature": 11977, "row_file": "rows/edit_011.npy", "row_sha256": "8577d277eea6b494dc60e02dc4bf6b2ddba329d36370df56e66fb3cf095cd41f"},
    # edit 12, feature 13643:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 12, "feature": 13643, "row_file": "rows/edit_012.npy", "row_sha256": "381af2f1e6abcbe36a61ccb270384d12a30ba0f311ce00485f203c422aeefa57"},
    # edit 13, feature 8278:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 13, "feature": 8278, "row_file": "rows/edit_013.npy", "row_sha256": "2a7ff5a126b06a38ff8609f7f92ef486492807f403d7e8184d7268d833a32f17"},
    # edit 14, feature 11218:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 14, "feature": 11218, "row_file": "rows/edit_014.npy", "row_sha256": "c6a9123a370c94bb5c8865b2de8ee309f1ba2e7059b0ffc68861272c5cfa31aa"},
    # edit 15, feature 125:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 15, "feature": 125, "row_file": "rows/edit_015.npy", "row_sha256": "7a79305e407c741aa0da4f25181eb40dad4e1a32a1ffd32e10e723f2bbb40ff7"},
    # edit 16, feature 10499:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 16, "feature": 10499, "row_file": "rows/edit_016.npy", "row_sha256": "f0c3b5121503df5a0bb52987ff150104ff4b82dfcf4363f9db6b8eb68926a24a"},
    # edit 17, feature 657:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 17, "feature": 657, "row_file": "rows/edit_017.npy", "row_sha256": "471266a0f8f7532a03346cb6cc5e96855dc6fb94ff8075813250b2c3d085b686"},
    # edit 18, feature 13887:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 18, "feature": 13887, "row_file": "rows/edit_018.npy", "row_sha256": "c57c254c14dfe79294f63ac719f9b2584663272760d51b14c0a792383ef725fc"},
    # edit 19, feature 263:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 19, "feature": 263, "row_file": "rows/edit_019.npy", "row_sha256": "de495e0f095ac7eb48fb0c85f7f6ab3b2b91f0f32ca747d6d043425ce538d712"},
    # edit 20, feature 13626:
    #   retention row rotated 0.32 of a unit step toward -d_hat, the source-balanced, mean-
    #   orthogonalized train class-mean adoption axis, so that u_A-u_R lies along the adopt/retain
    #   axis; length kept, cos with original 1/sqrt(1+a^2).
    {"edit": 20, "feature": 13626, "row_file": "rows/edit_020.npy", "row_sha256": "ca0ad5ab958962502492fd4ddd03b7ab981414c474272fba0b98e35eb7c4771e"},
]
