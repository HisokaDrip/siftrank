# Spotting port scans
A host touching many distinct destination ports in a short window is the classic sign of a vertical scan. Half-open SYN scans leave lots of SYNs with no completed handshake. Rate thresholds per source IP catch the noisy ones; slow scans need longer windows.
