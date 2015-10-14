Below is an example network adapter and camera configuration for a two-camera imaging system provided by [Point Grey](http://www.ptgrey.com/). It consists of two Blackfly GigE cameras (Point Grey part number: BFLY-PGE-09S2C) and a single dual-port POE GigE adapter card (Point Grey part number: GIGE-PCIE2-2P02).

###Maze Camera

- Adapter physical connection (looking at back of computer)
```
RJ45 ------------
          |      |
    L [  [ ]    [x]  ] R
```

- Adapter Settings
    - Model:         Intel 82574L Gigabit Network Connection
    - MAC:           00:B0:9D:DB:D9:63
    - MTU:           9000 
    - DHCP:          Disabled
    - IP:            192.168.0.100
    - Subnet mask:   255.255.255.0
- Camera Settings
    - Model:         Blackfly BFLY-PGE-09S2C
    - Serial No.:    14395177 
    - IP:            192.168.0.1 (Static)
    - Subnet mask:   255.255.255.0
    - Default GW:    0.0.0.0
    - Persistent IP: Yes
    
###Sleepbox Camera

- Adapter physical connection (looking at back of computer)
```
RJ45 ------------
          |      |
    L [  [x]    [ ]  ] R
```

- Adapter Settings
    - Model:         Intel 82574L Gigabit Network Connection
    - MAC:           00:B0:9D:DB:A7:29
    - MTU:           9000 
    - DHCP:          Disabled
    - IP:            192.168.1.100
    - Subnet mask:   255.255.255.0
- Camera Settings
    - Model:         Blackfly BFLY-PGE-09S2C
    - Serial No.:    14395177 
    - IP:            192.168.1.1 (Static)
    - Subnet mask:   255.255.255.0
    - Default GW:    0.0.0.0
    - Persistent IP: Yes