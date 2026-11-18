### E24r conclusion (Claude, 2026-11-02)
E24's hand head is not usable: against the calibrated FK hand point, error is ~130-140 px at t+10/20/30 and ~140 px
already at t+0 (encoding, not dynamics); copy error is 10-41 px; on moving windows prediction beats copy only 9-25%.
Overlays show predicted hand points on the forearms and sometimes on the wrong side: E24 learned the raw SAM3
hand_left/hand_right labels, which include forearms and switch sides (E22b-fix). E24's earlier hand numbers
(82-86 px vs raw SAM3) and the E24 object-loss contribution for hands are therefore unreliable.
Action: E25 Amendment 2 trains and evaluates hands on the cleaned hand-only masks, with the FK point as a check only.
The bottle labels come from the same SAM3 run (many identity switches in E21); a similar audit of bottle targets is advisable.
