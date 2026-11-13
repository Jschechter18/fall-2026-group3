"""Accepted hand edits of the decoder, in order. Replay with
mas_sae.llm_edit.edit_rules.read_edits_py and apply_edits onto the original SAE."""

EDITS = [
    # edit 1, feature 4409:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 1, "feature": 4409, "row_file": "rows/edit_001.npy", "row_sha256": "8072491134b5966fc733c3809a72e5957176484122f8e9498077b988c24148cc"},
    # edit 2, feature 12624:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 2, "feature": 12624, "row_file": "rows/edit_002.npy", "row_sha256": "0f2e4bd1c7a64ad84eda8aa71676662dd8cd6144e7231d258b8cb66d18eb1009"},
    # edit 3, feature 14120:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 3, "feature": 14120, "row_file": "rows/edit_003.npy", "row_sha256": "09b2c7812dc7ed58d3b97e49dd6806a8313b64b9461edf9139ceef7034df8208"},
    # edit 4, feature 10383:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 4, "feature": 10383, "row_file": "rows/edit_004.npy", "row_sha256": "79ef33ce2aa13285a9770ed60f44f2b7701115dd5f739709c852c5681ad0eba4"},
    # edit 5, feature 12139:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 5, "feature": 12139, "row_file": "rows/edit_005.npy", "row_sha256": "f9791f4ee6e53166c2834612304bbf1cd7697ee5d58b4c5db5374ea8d549de36"},
    # edit 6, feature 1750:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 6, "feature": 1750, "row_file": "rows/edit_006.npy", "row_sha256": "d0d68ce773239b0f99463cdbe75705c0dbcce5a9906b5c050272b00e1512bf97"},
    # edit 7, feature 16138:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 7, "feature": 16138, "row_file": "rows/edit_007.npy", "row_sha256": "924e6ba793308f3c9782ab45c9663a350e731bc19b8326bfd0b50f681f0a4574"},
    # edit 8, feature 16144:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 8, "feature": 16144, "row_file": "rows/edit_008.npy", "row_sha256": "c8a27560ad01a5d47416ed730de472e0d34738121fe7e562a881fa6c245fcc29"},
    # edit 9, feature 7649:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 9, "feature": 7649, "row_file": "rows/edit_009.npy", "row_sha256": "887f5af3ea8cab722a2c5f7ad6d1a993014c57a7ad8d05ce37f2eb34e46b5c2c"},
    # edit 10, feature 15005:
    #   rotate 17.3 deg toward normalize(behavior_axis - u_A): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 10, "feature": 15005, "row_file": "rows/edit_010.npy", "row_sha256": "c460d4424d2717b01fc672eb2fbd6903d983aab851315589369b781291953b08"},
    # edit 11, feature 11977:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 11, "feature": 11977, "row_file": "rows/edit_011.npy", "row_sha256": "c92a21b406b8d3172df86f98547b5a9ae3eba0f1bb4d3952131f80178fbd1864"},
    # edit 12, feature 13643:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 12, "feature": 13643, "row_file": "rows/edit_012.npy", "row_sha256": "f9017fec9976c3425d0e81e76b775f5f4ca025346e96bda41cf738a5b99c05f3"},
    # edit 13, feature 8278:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 13, "feature": 8278, "row_file": "rows/edit_013.npy", "row_sha256": "ea6dabff9fa9b9ad8fe02391d050bcd241bdf76e8984fe2b72211f96c22518a7"},
    # edit 14, feature 11218:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 14, "feature": 11218, "row_file": "rows/edit_014.npy", "row_sha256": "1b8a9a1b1ab989aa9d7c0a868643ac65c6f62dd3ab0f006751ad63436c8bed69"},
    # edit 15, feature 125:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 15, "feature": 125, "row_file": "rows/edit_015.npy", "row_sha256": "af662141b97998a2f0cf71814997ee5c20c93ddf71c8b8c6d3353836db70484c"},
    # edit 16, feature 10499:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 16, "feature": 10499, "row_file": "rows/edit_016.npy", "row_sha256": "fb9866b8b6162d50facd7ceb16bd0ba49c7206fe1a4972d169734cd8db808d05"},
    # edit 17, feature 657:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 17, "feature": 657, "row_file": "rows/edit_017.npy", "row_sha256": "653156ca323f109a206eba4aff0f350269f9c92801e2818a042e04ee764f51eb"},
    # edit 18, feature 13887:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 18, "feature": 13887, "row_file": "rows/edit_018.npy", "row_sha256": "c34ed6fcf3faeb18ef86402837cd34f35302e1599b18e18d1be97540ac426372"},
    # edit 19, feature 263:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 19, "feature": 263, "row_file": "rows/edit_019.npy", "row_sha256": "2a3bc2f829d9d4c5da0114dc19733d724654a9ea20fda1c9652591e2fdc69051"},
    # edit 20, feature 13626:
    #   rotate 17.3 deg toward normalize(-behavior_axis - u_R): behavior_axis is the train class-
    #   mean difference (adopt-retain, mean-activation axis removed), equal mix of the axis shared
    #   by musique/refusal cells and the axis of the majority synthetic-noref cell; the -u_A term
    #   cancels the set's old non-behavior direction coherently, so u_A-u_R aligns 0.96 with the
    #   behavior axis.
    {"edit": 20, "feature": 13626, "row_file": "rows/edit_020.npy", "row_sha256": "550253c1652fe0c1298d56448840d2ed581c66a6088c2d2082699632aba6d330"},
    # edit 21, feature 4409:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 21, "feature": 4409, "row_file": "rows/edit_021.npy", "row_sha256": "8a4ae0d11b9fb00b9fd929162c4d9e24fc01791329939c92019ab9c89efce6be"},
    # edit 22, feature 12624:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 22, "feature": 12624, "row_file": "rows/edit_022.npy", "row_sha256": "0b255167b02c37257fb0eec104d9a03222d4b33e336341670f13f0c411815cb2"},
    # edit 23, feature 14120:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 23, "feature": 14120, "row_file": "rows/edit_023.npy", "row_sha256": "fcf4680ed82128d238c770fedc935e1fff0a0fdbc4859e7864307514a9ab3754"},
    # edit 24, feature 10383:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 24, "feature": 10383, "row_file": "rows/edit_024.npy", "row_sha256": "63b57acf340d9a0dfa340b3168d5072e78ab8e216cb29b33a6477de47971cb1e"},
    # edit 25, feature 12139:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 25, "feature": 12139, "row_file": "rows/edit_025.npy", "row_sha256": "ea52610735b6bf21da5e7088a5aa82ae75bfd5b9c9e1086b05633479959cb54e"},
    # edit 26, feature 1750:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 26, "feature": 1750, "row_file": "rows/edit_026.npy", "row_sha256": "6c5566fe992f6a0e7a934d25a41b25dc0fcd8543fd591e950d77c1f980ff0eac"},
    # edit 27, feature 16138:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 27, "feature": 16138, "row_file": "rows/edit_027.npy", "row_sha256": "92b3f2f69dfe29dfef764eecefdd41c5045e659614658a02d4d853a77a747a86"},
    # edit 28, feature 16144:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 28, "feature": 16144, "row_file": "rows/edit_028.npy", "row_sha256": "c5001f5b6cd4d4eccc07e48aab10712624d201e7e5b46e88664b3f1a8d106b13"},
    # edit 29, feature 7649:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 29, "feature": 7649, "row_file": "rows/edit_029.npy", "row_sha256": "aaa201292f66fdea95a9a0f6cd42af05e99e5d3afc3262ca0df77c5f5daa30e5"},
    # edit 30, feature 15005:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 30, "feature": 15005, "row_file": "rows/edit_030.npy", "row_sha256": "af4a15b23c5d40b2f51386019734984036ff6e8932751b5db392a032cdc23973"},
    # edit 31, feature 11977:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 31, "feature": 11977, "row_file": "rows/edit_031.npy", "row_sha256": "4cb08a9c00adcb4fd6c22df3d3f93d615ee7da15a28b19c207f1842eac4fa0c8"},
    # edit 32, feature 13643:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 32, "feature": 13643, "row_file": "rows/edit_032.npy", "row_sha256": "463f47047289aa459a183aa643b5da8881cb1940984194ac101dea70c5db78b0"},
    # edit 33, feature 8278:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 33, "feature": 8278, "row_file": "rows/edit_033.npy", "row_sha256": "1a91654e8a45fd9b5cb0af4194e470c5d2e2a88e02e65f87a2a72233ab3e6cfe"},
    # edit 34, feature 11218:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 34, "feature": 11218, "row_file": "rows/edit_034.npy", "row_sha256": "b8480de8f64c83095b3ba04d55900fe8c63e01815b128a6b7db414c46b19f4e5"},
    # edit 35, feature 125:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 35, "feature": 125, "row_file": "rows/edit_035.npy", "row_sha256": "e8861c3f02418483b020d9b920603ec5137915888e5e7909941a4dad50a135c2"},
    # edit 36, feature 10499:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 36, "feature": 10499, "row_file": "rows/edit_036.npy", "row_sha256": "1ae44d010d58debe1a791326953278ad10619096e375f1f92d3cc00795ace31c"},
    # edit 37, feature 657:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 37, "feature": 657, "row_file": "rows/edit_037.npy", "row_sha256": "9564fbb7cb19040819e0443e1b788ce8f3f59d18457d4cd049be0a44163a07a2"},
    # edit 38, feature 13887:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 38, "feature": 13887, "row_file": "rows/edit_038.npy", "row_sha256": "c01f6d71a3b0e00b379f3c479d210e4fb0ef5622b57f6a4395429c4574d8e944"},
    # edit 39, feature 263:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 39, "feature": 263, "row_file": "rows/edit_039.npy", "row_sha256": "7c1b85afe8dd0f3fd085afa3557d7d32c5b37f05773b13ce7e39aa836b05d489"},
    # edit 40, feature 13626:
    #   dial the same edit down: each row rotated only 9.2 deg toward +/-behavior_axis (same axis
    #   as before: train class-mean difference adopt-retain, mean-activation axis removed, equal
    #   mix of the small-cell axis and the majority-cell axis). The steering direction is then
    #   0.67*behavior_axis + 0.81*original direction; the original direction is near-inert
    #   (baseline margin -1.1), so this gives ~70% of the behavior push and should derail fewer
    #   generations into third answers.
    {"edit": 40, "feature": 13626, "row_file": "rows/edit_040.npy", "row_sha256": "05d5636651ccf3fbbd6d67ad1c7d3cfcb42ee5d24a0572be192d4804b9a51567"},
    # edit 41, feature 4409:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 41, "feature": 4409, "row_file": "rows/edit_041.npy", "row_sha256": "8072491134b5966fc733c3809a72e5957176484122f8e9498077b988c24148cc"},
    # edit 42, feature 12624:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 42, "feature": 12624, "row_file": "rows/edit_042.npy", "row_sha256": "0f2e4bd1c7a64ad84eda8aa71676662dd8cd6144e7231d258b8cb66d18eb1009"},
    # edit 43, feature 14120:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 43, "feature": 14120, "row_file": "rows/edit_043.npy", "row_sha256": "09b2c7812dc7ed58d3b97e49dd6806a8313b64b9461edf9139ceef7034df8208"},
    # edit 44, feature 10383:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 44, "feature": 10383, "row_file": "rows/edit_044.npy", "row_sha256": "79ef33ce2aa13285a9770ed60f44f2b7701115dd5f739709c852c5681ad0eba4"},
    # edit 45, feature 12139:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 45, "feature": 12139, "row_file": "rows/edit_045.npy", "row_sha256": "f9791f4ee6e53166c2834612304bbf1cd7697ee5d58b4c5db5374ea8d549de36"},
    # edit 46, feature 1750:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 46, "feature": 1750, "row_file": "rows/edit_046.npy", "row_sha256": "d0d68ce773239b0f99463cdbe75705c0dbcce5a9906b5c050272b00e1512bf97"},
    # edit 47, feature 16138:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 47, "feature": 16138, "row_file": "rows/edit_047.npy", "row_sha256": "924e6ba793308f3c9782ab45c9663a350e731bc19b8326bfd0b50f681f0a4574"},
    # edit 48, feature 16144:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 48, "feature": 16144, "row_file": "rows/edit_048.npy", "row_sha256": "c8a27560ad01a5d47416ed730de472e0d34738121fe7e562a881fa6c245fcc29"},
    # edit 49, feature 7649:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 49, "feature": 7649, "row_file": "rows/edit_049.npy", "row_sha256": "887f5af3ea8cab722a2c5f7ad6d1a993014c57a7ad8d05ce37f2eb34e46b5c2c"},
    # edit 50, feature 15005:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 50, "feature": 15005, "row_file": "rows/edit_050.npy", "row_sha256": "c460d4424d2717b01fc672eb2fbd6903d983aab851315589369b781291953b08"},
    # edit 51, feature 11977:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 51, "feature": 11977, "row_file": "rows/edit_051.npy", "row_sha256": "c92a21b406b8d3172df86f98547b5a9ae3eba0f1bb4d3952131f80178fbd1864"},
    # edit 52, feature 13643:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 52, "feature": 13643, "row_file": "rows/edit_052.npy", "row_sha256": "f9017fec9976c3425d0e81e76b775f5f4ca025346e96bda41cf738a5b99c05f3"},
    # edit 53, feature 8278:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 53, "feature": 8278, "row_file": "rows/edit_053.npy", "row_sha256": "ea6dabff9fa9b9ad8fe02391d050bcd241bdf76e8984fe2b72211f96c22518a7"},
    # edit 54, feature 11218:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 54, "feature": 11218, "row_file": "rows/edit_054.npy", "row_sha256": "1b8a9a1b1ab989aa9d7c0a868643ac65c6f62dd3ab0f006751ad63436c8bed69"},
    # edit 55, feature 125:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 55, "feature": 125, "row_file": "rows/edit_055.npy", "row_sha256": "af662141b97998a2f0cf71814997ee5c20c93ddf71c8b8c6d3353836db70484c"},
    # edit 56, feature 10499:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 56, "feature": 10499, "row_file": "rows/edit_056.npy", "row_sha256": "fb9866b8b6162d50facd7ceb16bd0ba49c7206fe1a4972d169734cd8db808d05"},
    # edit 57, feature 657:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 57, "feature": 657, "row_file": "rows/edit_057.npy", "row_sha256": "653156ca323f109a206eba4aff0f350269f9c92801e2818a042e04ee764f51eb"},
    # edit 58, feature 13887:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 58, "feature": 13887, "row_file": "rows/edit_058.npy", "row_sha256": "c34ed6fcf3faeb18ef86402837cd34f35302e1599b18e18d1be97540ac426372"},
    # edit 59, feature 263:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 59, "feature": 263, "row_file": "rows/edit_059.npy", "row_sha256": "2a3bc2f829d9d4c5da0114dc19733d724654a9ea20fda1c9652591e2fdc69051"},
    # edit 60, feature 13626:
    #   restore full-strength edit: each row rotated 17.3 deg (cos 0.955) toward
    #   normalize(+/-behavior_axis - u_set). behavior_axis = train class-mean difference
    #   (adoption-retention) with the mean-activation axis removed, as an equal mix of two near-
    #   orthogonal measured axes: the axis shared by the musique and critic-refusal cells and the
    #   axis of the majority synthetic/non-refusal cell. The -u_set term coherently cancels each
    #   set's old (non-behavior) direction, so normalize(u_A-u_R) has cosine 0.959 with the
    #   behavior axis instead of 0.11. Test showed 70% strength halves the effect, so strength is
    #   kept maximal.
    {"edit": 60, "feature": 13626, "row_file": "rows/edit_060.npy", "row_sha256": "550253c1652fe0c1298d56448840d2ed581c66a6088c2d2082699632aba6d330"},
]
