
bin/seq_bench/gcc/matmul_O3_native_unroll:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	48 83 ec 08          	sub    $0x8,%rsp
    1004:	48 8b 05 c5 2f 00 00 	mov    0x2fc5(%rip),%rax        # 3fd0 <__gmon_start__@Base>
    100b:	48 85 c0             	test   %rax,%rax
    100e:	74 02                	je     1012 <_init+0x12>
    1010:	ff d0                	call   *%rax
    1012:	48 83 c4 08          	add    $0x8,%rsp
    1016:	c3                   	ret

Disassembly of section .plt:

0000000000001020 <free@plt-0x10>:
    1020:	ff 35 ca 2f 00 00    	push   0x2fca(%rip)        # 3ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 cc 2f 00 00    	jmp    *0x2fcc(%rip)        # 3ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001030 <free@plt>:
    1030:	ff 25 ca 2f 00 00    	jmp    *0x2fca(%rip)        # 4000 <free@GLIBC_2.2.5>
    1036:	68 00 00 00 00       	push   $0x0
    103b:	e9 e0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001040 <clock@plt>:
    1040:	ff 25 c2 2f 00 00    	jmp    *0x2fc2(%rip)        # 4008 <clock@GLIBC_2.2.5>
    1046:	68 01 00 00 00       	push   $0x1
    104b:	e9 d0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001050 <fclose@plt>:
    1050:	ff 25 ba 2f 00 00    	jmp    *0x2fba(%rip)        # 4010 <fclose@GLIBC_2.2.5>
    1056:	68 02 00 00 00       	push   $0x2
    105b:	e9 c0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001060 <fputc@plt>:
    1060:	ff 25 b2 2f 00 00    	jmp    *0x2fb2(%rip)        # 4018 <fputc@GLIBC_2.2.5>
    1066:	68 03 00 00 00       	push   $0x3
    106b:	e9 b0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001070 <calloc@plt>:
    1070:	ff 25 aa 2f 00 00    	jmp    *0x2faa(%rip)        # 4020 <calloc@GLIBC_2.2.5>
    1076:	68 04 00 00 00       	push   $0x4
    107b:	e9 a0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001080 <fprintf@plt>:
    1080:	ff 25 a2 2f 00 00    	jmp    *0x2fa2(%rip)        # 4028 <fprintf@GLIBC_2.2.5>
    1086:	68 05 00 00 00       	push   $0x5
    108b:	e9 90 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001090 <malloc@plt>:
    1090:	ff 25 9a 2f 00 00    	jmp    *0x2f9a(%rip)        # 4030 <malloc@GLIBC_2.2.5>
    1096:	68 06 00 00 00       	push   $0x6
    109b:	e9 80 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010a0 <fopen@plt>:
    10a0:	ff 25 92 2f 00 00    	jmp    *0x2f92(%rip)        # 4038 <fopen@GLIBC_2.2.5>
    10a6:	68 07 00 00 00       	push   $0x7
    10ab:	e9 70 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010b0 <perror@plt>:
    10b0:	ff 25 8a 2f 00 00    	jmp    *0x2f8a(%rip)        # 4040 <perror@GLIBC_2.2.5>
    10b6:	68 08 00 00 00       	push   $0x8
    10bb:	e9 60 ff ff ff       	jmp    1020 <_init+0x20>

Disassembly of section .plt.got:

00000000000010c0 <__cxa_finalize@plt>:
    10c0:	ff 25 1a 2f 00 00    	jmp    *0x2f1a(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    10c6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000010d0 <main>:
    10d0:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    10d5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    10d9:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    10de:	41 ff 72 f8          	push   -0x8(%r10)
    10e2:	55                   	push   %rbp
    10e3:	48 89 e5             	mov    %rsp,%rbp
    10e6:	41 57                	push   %r15
    10e8:	41 56                	push   %r14
    10ea:	41 55                	push   %r13
    10ec:	41 54                	push   %r12
    10ee:	41 52                	push   %r10
    10f0:	53                   	push   %rbx
    10f1:	48 83 ec 20          	sub    $0x20,%rsp
    10f5:	e8 96 ff ff ff       	call   1090 <malloc@plt>
    10fa:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    10ff:	49 89 c7             	mov    %rax,%r15
    1102:	e8 89 ff ff ff       	call   1090 <malloc@plt>
    1107:	be 01 00 00 00       	mov    $0x1,%esi
    110c:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    1111:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    1115:	e8 56 ff ff ff       	call   1070 <calloc@plt>
    111a:	4c 89 f9             	mov    %r15,%rcx
    111d:	c4 e2 7d 19 05 1a 0f 	vbroadcastsd 0xf1a(%rip),%ymm0        # 2040 <_IO_stdin_used+0x40>
    1124:	00 00 
    1126:	49 8d b7 40 9c 00 00 	lea    0x9c40(%r15),%rsi
    112d:	49 89 c5             	mov    %rax,%r13
    1130:	49 8d bf 40 5e ec 0b 	lea    0xbec5e40(%r15),%rdi
    1137:	48 89 f2             	mov    %rsi,%rdx
    113a:	48 89 c8             	mov    %rcx,%rax
    113d:	48 29 ca             	sub    %rcx,%rdx
    1140:	48 83 ea 20          	sub    $0x20,%rdx
    1144:	48 c1 ea 05          	shr    $0x5,%rdx
    1148:	48 83 c2 01          	add    $0x1,%rdx
    114c:	83 e2 07             	and    $0x7,%edx
    114f:	74 61                	je     11b2 <main+0xe2>
    1151:	48 83 fa 01          	cmp    $0x1,%rdx
    1155:	74 4e                	je     11a5 <main+0xd5>
    1157:	48 83 fa 02          	cmp    $0x2,%rdx
    115b:	74 40                	je     119d <main+0xcd>
    115d:	48 83 fa 03          	cmp    $0x3,%rdx
    1161:	74 32                	je     1195 <main+0xc5>
    1163:	48 83 fa 04          	cmp    $0x4,%rdx
    1167:	74 24                	je     118d <main+0xbd>
    1169:	48 83 fa 05          	cmp    $0x5,%rdx
    116d:	74 16                	je     1185 <main+0xb5>
    116f:	48 83 fa 06          	cmp    $0x6,%rdx
    1173:	74 08                	je     117d <main+0xad>
    1175:	c5 fd 11 01          	vmovupd %ymm0,(%rcx)
    1179:	48 8d 41 20          	lea    0x20(%rcx),%rax
    117d:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    1181:	48 83 c0 20          	add    $0x20,%rax
    1185:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    1189:	48 83 c0 20          	add    $0x20,%rax
    118d:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    1191:	48 83 c0 20          	add    $0x20,%rax
    1195:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    1199:	48 83 c0 20          	add    $0x20,%rax
    119d:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    11a1:	48 83 c0 20          	add    $0x20,%rax
    11a5:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    11a9:	48 83 c0 20          	add    $0x20,%rax
    11ad:	48 39 f0             	cmp    %rsi,%rax
    11b0:	74 3b                	je     11ed <main+0x11d>
    11b2:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    11b6:	48 05 00 01 00 00    	add    $0x100,%rax
    11bc:	c5 fd 11 80 20 ff ff 	vmovupd %ymm0,-0xe0(%rax)
    11c3:	ff 
    11c4:	c5 fd 11 80 40 ff ff 	vmovupd %ymm0,-0xc0(%rax)
    11cb:	ff 
    11cc:	c5 fd 11 80 60 ff ff 	vmovupd %ymm0,-0xa0(%rax)
    11d3:	ff 
    11d4:	c5 fd 11 40 80       	vmovupd %ymm0,-0x80(%rax)
    11d9:	c5 fd 11 40 a0       	vmovupd %ymm0,-0x60(%rax)
    11de:	c5 fd 11 40 c0       	vmovupd %ymm0,-0x40(%rax)
    11e3:	c5 fd 11 40 e0       	vmovupd %ymm0,-0x20(%rax)
    11e8:	48 39 f0             	cmp    %rsi,%rax
    11eb:	75 c5                	jne    11b2 <main+0xe2>
    11ed:	48 8d b0 40 9c 00 00 	lea    0x9c40(%rax),%rsi
    11f4:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    11fb:	48 39 fe             	cmp    %rdi,%rsi
    11fe:	0f 85 33 ff ff ff    	jne    1137 <main+0x67>
    1204:	c4 e2 7d 19 0d 3b 0e 	vbroadcastsd 0xe3b(%rip),%ymm1        # 2048 <_IO_stdin_used+0x48>
    120b:	00 00 
    120d:	48 8b 5d c8          	mov    -0x38(%rbp),%rbx
    1211:	4c 8d 83 40 9c 00 00 	lea    0x9c40(%rbx),%r8
    1218:	4c 8d 8b 40 5e ec 0b 	lea    0xbec5e40(%rbx),%r9
    121f:	4d 89 c3             	mov    %r8,%r11
    1222:	49 89 dc             	mov    %rbx,%r12
    1225:	49 29 db             	sub    %rbx,%r11
    1228:	49 83 eb 20          	sub    $0x20,%r11
    122c:	49 c1 eb 05          	shr    $0x5,%r11
    1230:	49 83 c3 01          	add    $0x1,%r11
    1234:	41 83 e3 07          	and    $0x7,%r11d
    1238:	74 6d                	je     12a7 <main+0x1d7>
    123a:	49 83 fb 01          	cmp    $0x1,%r11
    123e:	74 58                	je     1298 <main+0x1c8>
    1240:	49 83 fb 02          	cmp    $0x2,%r11
    1244:	74 48                	je     128e <main+0x1be>
    1246:	49 83 fb 03          	cmp    $0x3,%r11
    124a:	74 38                	je     1284 <main+0x1b4>
    124c:	49 83 fb 04          	cmp    $0x4,%r11
    1250:	74 28                	je     127a <main+0x1aa>
    1252:	49 83 fb 05          	cmp    $0x5,%r11
    1256:	74 18                	je     1270 <main+0x1a0>
    1258:	49 83 fb 06          	cmp    $0x6,%r11
    125c:	74 08                	je     1266 <main+0x196>
    125e:	c5 fd 11 0b          	vmovupd %ymm1,(%rbx)
    1262:	4c 8d 63 20          	lea    0x20(%rbx),%r12
    1266:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    126c:	49 83 c4 20          	add    $0x20,%r12
    1270:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    1276:	49 83 c4 20          	add    $0x20,%r12
    127a:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    1280:	49 83 c4 20          	add    $0x20,%r12
    1284:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    128a:	49 83 c4 20          	add    $0x20,%r12
    128e:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    1294:	49 83 c4 20          	add    $0x20,%r12
    1298:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    129e:	49 83 c4 20          	add    $0x20,%r12
    12a2:	4d 39 e0             	cmp    %r12,%r8
    12a5:	74 4c                	je     12f3 <main+0x223>
    12a7:	c4 c1 7d 11 0c 24    	vmovupd %ymm1,(%r12)
    12ad:	49 81 c4 00 01 00 00 	add    $0x100,%r12
    12b4:	c4 c1 7d 11 8c 24 20 	vmovupd %ymm1,-0xe0(%r12)
    12bb:	ff ff ff 
    12be:	c4 c1 7d 11 8c 24 40 	vmovupd %ymm1,-0xc0(%r12)
    12c5:	ff ff ff 
    12c8:	c4 c1 7d 11 8c 24 60 	vmovupd %ymm1,-0xa0(%r12)
    12cf:	ff ff ff 
    12d2:	c4 c1 7d 11 4c 24 80 	vmovupd %ymm1,-0x80(%r12)
    12d9:	c4 c1 7d 11 4c 24 a0 	vmovupd %ymm1,-0x60(%r12)
    12e0:	c4 c1 7d 11 4c 24 c0 	vmovupd %ymm1,-0x40(%r12)
    12e7:	c4 c1 7d 11 4c 24 e0 	vmovupd %ymm1,-0x20(%r12)
    12ee:	4d 39 e0             	cmp    %r12,%r8
    12f1:	75 b4                	jne    12a7 <main+0x1d7>
    12f3:	49 81 c0 40 9c 00 00 	add    $0x9c40,%r8
    12fa:	48 81 c3 40 9c 00 00 	add    $0x9c40,%rbx
    1301:	4d 39 c8             	cmp    %r9,%r8
    1304:	0f 85 15 ff ff ff    	jne    121f <main+0x14f>
    130a:	c5 f8 77             	vzeroupper
    130d:	e8 2e fd ff ff       	call   1040 <clock@plt>
    1312:	4d 8d 57 08          	lea    0x8(%r15),%r10
    1316:	4d 89 e8             	mov    %r13,%r8
    1319:	48 89 c3             	mov    %rax,%rbx
    131c:	4d 8d 8d 00 c2 eb 0b 	lea    0xbebc200(%r13),%r9
    1323:	48 8b 4d c8          	mov    -0x38(%rbp),%rcx
    1327:	4d 89 d6             	mov    %r10,%r14
    132a:	bf 01 00 00 00       	mov    $0x1,%edi
    132f:	c4 c2 7d 19 6e f8    	vbroadcastsd -0x8(%r14),%ymm5
    1335:	c4 c2 7d 19 36       	vbroadcastsd (%r14),%ymm6
    133a:	4c 89 c2             	mov    %r8,%rdx
    133d:	31 c0                	xor    %eax,%eax
    133f:	48 8d b1 40 9c 00 00 	lea    0x9c40(%rcx),%rsi
    1346:	c5 d5 59 54 08 20    	vmulpd 0x20(%rax,%rcx,1),%ymm5,%ymm2
    134c:	48 81 c2 40 01 00 00 	add    $0x140,%rdx
    1353:	c5 ed 58 a2 e0 fe ff 	vaddpd -0x120(%rdx),%ymm2,%ymm4
    135a:	ff 
    135b:	c5 cd 59 5c 06 20    	vmulpd 0x20(%rsi,%rax,1),%ymm6,%ymm3
    1361:	c5 cd 59 54 06 40    	vmulpd 0x40(%rsi,%rax,1),%ymm6,%ymm2
    1367:	c5 d5 59 44 08 40    	vmulpd 0x40(%rax,%rcx,1),%ymm5,%ymm0
    136d:	c5 fd 58 8a 00 ff ff 	vaddpd -0x100(%rdx),%ymm0,%ymm1
    1374:	ff 
    1375:	c5 4d 59 14 06       	vmulpd (%rsi,%rax,1),%ymm6,%ymm10
    137a:	c5 4d 59 74 06 60    	vmulpd 0x60(%rsi,%rax,1),%ymm6,%ymm14
    1380:	c5 55 59 64 08 60    	vmulpd 0x60(%rax,%rcx,1),%ymm5,%ymm12
    1386:	c5 dd 58 fb          	vaddpd %ymm3,%ymm4,%ymm7
    138a:	c5 55 59 04 01       	vmulpd (%rcx,%rax,1),%ymm5,%ymm8
    138f:	c5 f5 58 e2          	vaddpd %ymm2,%ymm1,%ymm4
    1393:	c5 1d 58 aa 20 ff ff 	vaddpd -0xe0(%rdx),%ymm12,%ymm13
    139a:	ff 
    139b:	c5 3d 58 8a c0 fe ff 	vaddpd -0x140(%rdx),%ymm8,%ymm9
    13a2:	ff 
    13a3:	c5 4d 59 a4 06 80 00 	vmulpd 0x80(%rsi,%rax,1),%ymm6,%ymm12
    13aa:	00 00 
    13ac:	c5 4d 59 84 06 a0 00 	vmulpd 0xa0(%rsi,%rax,1),%ymm6,%ymm8
    13b3:	00 00 
    13b5:	c5 fd 11 ba e0 fe ff 	vmovupd %ymm7,-0x120(%rdx)
    13bc:	ff 
    13bd:	c4 41 15 58 fe       	vaddpd %ymm14,%ymm13,%ymm15
    13c2:	c5 fd 11 a2 00 ff ff 	vmovupd %ymm4,-0x100(%rdx)
    13c9:	ff 
    13ca:	c5 cd 59 84 06 e0 00 	vmulpd 0xe0(%rsi,%rax,1),%ymm6,%ymm0
    13d1:	00 00 
    13d3:	c5 cd 59 a4 06 c0 00 	vmulpd 0xc0(%rsi,%rax,1),%ymm6,%ymm4
    13da:	00 00 
    13dc:	c4 41 35 58 da       	vaddpd %ymm10,%ymm9,%ymm11
    13e1:	c5 d5 59 9c 08 a0 00 	vmulpd 0xa0(%rax,%rcx,1),%ymm5,%ymm3
    13e8:	00 00 
    13ea:	c5 e5 58 ba 60 ff ff 	vaddpd -0xa0(%rdx),%ymm3,%ymm7
    13f1:	ff 
    13f2:	c5 7d 11 ba 20 ff ff 	vmovupd %ymm15,-0xe0(%rdx)
    13f9:	ff 
    13fa:	c5 55 59 94 08 80 00 	vmulpd 0x80(%rax,%rcx,1),%ymm5,%ymm10
    1401:	00 00 
    1403:	c5 55 59 b4 08 e0 00 	vmulpd 0xe0(%rax,%rcx,1),%ymm5,%ymm14
    140a:	00 00 
    140c:	c5 7d 11 9a c0 fe ff 	vmovupd %ymm11,-0x140(%rdx)
    1413:	ff 
    1414:	c5 2d 58 9a 40 ff ff 	vaddpd -0xc0(%rdx),%ymm10,%ymm11
    141b:	ff 
    141c:	c5 d5 59 94 08 c0 00 	vmulpd 0xc0(%rax,%rcx,1),%ymm5,%ymm2
    1423:	00 00 
    1425:	c5 0d 58 7a a0       	vaddpd -0x60(%rdx),%ymm14,%ymm15
    142a:	c5 ed 58 5a 80       	vaddpd -0x80(%rdx),%ymm2,%ymm3
    142f:	c4 41 45 58 c8       	vaddpd %ymm8,%ymm7,%ymm9
    1434:	c4 41 25 58 ec       	vaddpd %ymm12,%ymm11,%ymm13
    1439:	c5 85 58 c8          	vaddpd %ymm0,%ymm15,%ymm1
    143d:	c5 e5 58 fc          	vaddpd %ymm4,%ymm3,%ymm7
    1441:	c5 7d 11 8a 60 ff ff 	vmovupd %ymm9,-0xa0(%rdx)
    1448:	ff 
    1449:	c5 7d 11 aa 40 ff ff 	vmovupd %ymm13,-0xc0(%rdx)
    1450:	ff 
    1451:	c5 fd 11 4a a0       	vmovupd %ymm1,-0x60(%rdx)
    1456:	c5 fd 11 7a 80       	vmovupd %ymm7,-0x80(%rdx)
    145b:	c5 55 59 84 08 20 01 	vmulpd 0x120(%rax,%rcx,1),%ymm5,%ymm8
    1462:	00 00 
    1464:	c5 4d 59 94 06 20 01 	vmulpd 0x120(%rsi,%rax,1),%ymm6,%ymm10
    146b:	00 00 
    146d:	c5 3d 58 4a e0       	vaddpd -0x20(%rdx),%ymm8,%ymm9
    1472:	c5 4d 59 b4 06 00 01 	vmulpd 0x100(%rsi,%rax,1),%ymm6,%ymm14
    1479:	00 00 
    147b:	48 05 40 01 00 00    	add    $0x140,%rax
    1481:	c5 55 59 64 08 c0    	vmulpd -0x40(%rax,%rcx,1),%ymm5,%ymm12
    1487:	c5 1d 58 6a c0       	vaddpd -0x40(%rdx),%ymm12,%ymm13
    148c:	c4 41 35 58 da       	vaddpd %ymm10,%ymm9,%ymm11
    1491:	c4 41 15 58 fe       	vaddpd %ymm14,%ymm13,%ymm15
    1496:	c5 7d 11 5a e0       	vmovupd %ymm11,-0x20(%rdx)
    149b:	c5 7d 11 7a c0       	vmovupd %ymm15,-0x40(%rdx)
    14a0:	48 3d 40 9c 00 00    	cmp    $0x9c40,%rax
    14a6:	0f 85 9a fe ff ff    	jne    1346 <main+0x276>
    14ac:	83 c7 02             	add    $0x2,%edi
    14af:	48 81 c1 80 38 01 00 	add    $0x13880,%rcx
    14b6:	49 83 c6 10          	add    $0x10,%r14
    14ba:	81 ff 89 13 00 00    	cmp    $0x1389,%edi
    14c0:	0f 85 69 fe ff ff    	jne    132f <main+0x25f>
    14c6:	49 81 c0 40 9c 00 00 	add    $0x9c40,%r8
    14cd:	49 81 c2 40 9c 00 00 	add    $0x9c40,%r10
    14d4:	4d 39 c1             	cmp    %r8,%r9
    14d7:	0f 85 46 fe ff ff    	jne    1323 <main+0x253>
    14dd:	c5 f8 77             	vzeroupper
    14e0:	e8 5b fb ff ff       	call   1040 <clock@plt>
    14e5:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    14e9:	48 8b 3d 70 2b 00 00 	mov    0x2b70(%rip),%rdi        # 4060 <stderr@GLIBC_2.2.5>
    14f0:	48 29 d8             	sub    %rbx,%rax
    14f3:	ba 88 13 00 00       	mov    $0x1388,%edx
    14f8:	48 8d 35 05 0b 00 00 	lea    0xb05(%rip),%rsi        # 2004 <_IO_stdin_used+0x4>
    14ff:	c4 e1 d3 2a f0       	vcvtsi2sd %rax,%xmm5,%xmm6
    1504:	b8 01 00 00 00       	mov    $0x1,%eax
    1509:	c5 cb 5e 05 3f 0b 00 	vdivsd 0xb3f(%rip),%xmm6,%xmm0        # 2050 <_IO_stdin_used+0x50>
    1510:	00 
    1511:	e8 6a fb ff ff       	call   1080 <fprintf@plt>
    1516:	48 8d 35 02 0b 00 00 	lea    0xb02(%rip),%rsi        # 201f <_IO_stdin_used+0x1f>
    151d:	48 8d 3d fd 0a 00 00 	lea    0xafd(%rip),%rdi        # 2021 <_IO_stdin_used+0x21>
    1524:	e8 77 fb ff ff       	call   10a0 <fopen@plt>
    1529:	48 85 c0             	test   %rax,%rax
    152c:	48 89 c3             	mov    %rax,%rbx
    152f:	0f 84 52 01 00 00    	je     1687 <main+0x5b7>
    1535:	48 89 c7             	mov    %rax,%rdi
    1538:	ba 88 13 00 00       	mov    $0x1388,%edx
    153d:	31 c0                	xor    %eax,%eax
    153f:	48 8d 35 ed 0a 00 00 	lea    0xaed(%rip),%rsi        # 2033 <_IO_stdin_used+0x33>
    1546:	4c 8d 25 eb 0a 00 00 	lea    0xaeb(%rip),%r12        # 2038 <_IO_stdin_used+0x38>
    154d:	e8 2e fb ff ff       	call   1080 <fprintf@plt>
    1552:	4d 8d 9d 40 79 62 02 	lea    0x2627940(%r13),%r11
    1559:	49 8d 8d 40 1f 00 00 	lea    0x1f40(%r13),%rcx
    1560:	4c 89 5d c0          	mov    %r11,-0x40(%rbp)
    1564:	4c 8d b1 c0 e0 ff ff 	lea    -0x1f40(%rcx),%r14
    156b:	c4 c1 7b 10 06       	vmovsd (%r14),%xmm0
    1570:	4c 89 e6             	mov    %r12,%rsi
    1573:	48 89 df             	mov    %rbx,%rdi
    1576:	b8 01 00 00 00       	mov    $0x1,%eax
    157b:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
    157f:	49 83 c6 40          	add    $0x40,%r14
    1583:	e8 f8 fa ff ff       	call   1080 <fprintf@plt>
    1588:	4c 89 e6             	mov    %r12,%rsi
    158b:	48 89 df             	mov    %rbx,%rdi
    158e:	b8 01 00 00 00       	mov    $0x1,%eax
    1593:	c4 c1 7b 10 46 c8    	vmovsd -0x38(%r14),%xmm0
    1599:	e8 e2 fa ff ff       	call   1080 <fprintf@plt>
    159e:	4c 89 e6             	mov    %r12,%rsi
    15a1:	48 89 df             	mov    %rbx,%rdi
    15a4:	b8 01 00 00 00       	mov    $0x1,%eax
    15a9:	c4 c1 7b 10 46 d0    	vmovsd -0x30(%r14),%xmm0
    15af:	e8 cc fa ff ff       	call   1080 <fprintf@plt>
    15b4:	4c 89 e6             	mov    %r12,%rsi
    15b7:	48 89 df             	mov    %rbx,%rdi
    15ba:	b8 01 00 00 00       	mov    $0x1,%eax
    15bf:	c4 c1 7b 10 46 d8    	vmovsd -0x28(%r14),%xmm0
    15c5:	e8 b6 fa ff ff       	call   1080 <fprintf@plt>
    15ca:	4c 89 e6             	mov    %r12,%rsi
    15cd:	48 89 df             	mov    %rbx,%rdi
    15d0:	b8 01 00 00 00       	mov    $0x1,%eax
    15d5:	c4 c1 7b 10 46 e0    	vmovsd -0x20(%r14),%xmm0
    15db:	e8 a0 fa ff ff       	call   1080 <fprintf@plt>
    15e0:	4c 89 e6             	mov    %r12,%rsi
    15e3:	48 89 df             	mov    %rbx,%rdi
    15e6:	b8 01 00 00 00       	mov    $0x1,%eax
    15eb:	c4 c1 7b 10 46 e8    	vmovsd -0x18(%r14),%xmm0
    15f1:	e8 8a fa ff ff       	call   1080 <fprintf@plt>
    15f6:	4c 89 e6             	mov    %r12,%rsi
    15f9:	48 89 df             	mov    %rbx,%rdi
    15fc:	b8 01 00 00 00       	mov    $0x1,%eax
    1601:	c4 c1 7b 10 46 f0    	vmovsd -0x10(%r14),%xmm0
    1607:	e8 74 fa ff ff       	call   1080 <fprintf@plt>
    160c:	4c 89 e6             	mov    %r12,%rsi
    160f:	48 89 df             	mov    %rbx,%rdi
    1612:	b8 01 00 00 00       	mov    $0x1,%eax
    1617:	c4 c1 7b 10 46 f8    	vmovsd -0x8(%r14),%xmm0
    161d:	e8 5e fa ff ff       	call   1080 <fprintf@plt>
    1622:	48 8b 4d b8          	mov    -0x48(%rbp),%rcx
    1626:	49 39 ce             	cmp    %rcx,%r14
    1629:	0f 85 3c ff ff ff    	jne    156b <main+0x49b>
    162f:	bf 0a 00 00 00       	mov    $0xa,%edi
    1634:	48 89 de             	mov    %rbx,%rsi
    1637:	e8 24 fa ff ff       	call   1060 <fputc@plt>
    163c:	48 8b 7d c0          	mov    -0x40(%rbp),%rdi
    1640:	49 8d 8e 40 9c 00 00 	lea    0x9c40(%r14),%rcx
    1647:	48 39 f9             	cmp    %rdi,%rcx
    164a:	0f 85 14 ff ff ff    	jne    1564 <main+0x494>
    1650:	48 89 df             	mov    %rbx,%rdi
    1653:	e8 f8 f9 ff ff       	call   1050 <fclose@plt>
    1658:	4c 89 ff             	mov    %r15,%rdi
    165b:	e8 d0 f9 ff ff       	call   1030 <free@plt>
    1660:	48 8b 7d c8          	mov    -0x38(%rbp),%rdi
    1664:	e8 c7 f9 ff ff       	call   1030 <free@plt>
    1669:	4c 89 ef             	mov    %r13,%rdi
    166c:	e8 bf f9 ff ff       	call   1030 <free@plt>
    1671:	31 c0                	xor    %eax,%eax
    1673:	48 83 c4 20          	add    $0x20,%rsp
    1677:	5b                   	pop    %rbx
    1678:	5e                   	pop    %rsi
    1679:	41 5c                	pop    %r12
    167b:	41 5d                	pop    %r13
    167d:	41 5e                	pop    %r14
    167f:	41 5f                	pop    %r15
    1681:	5d                   	pop    %rbp
    1682:	48 8d 66 f8          	lea    -0x8(%rsi),%rsp
    1686:	c3                   	ret
    1687:	48 8d 3d 9f 09 00 00 	lea    0x99f(%rip),%rdi        # 202d <_IO_stdin_used+0x2d>
    168e:	e8 1d fa ff ff       	call   10b0 <perror@plt>
    1693:	b8 01 00 00 00       	mov    $0x1,%eax
    1698:	eb d9                	jmp    1673 <main+0x5a3>
    169a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000016a0 <_start>:
    16a0:	31 ed                	xor    %ebp,%ebp
    16a2:	49 89 d1             	mov    %rdx,%r9
    16a5:	5e                   	pop    %rsi
    16a6:	48 89 e2             	mov    %rsp,%rdx
    16a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    16ad:	50                   	push   %rax
    16ae:	54                   	push   %rsp
    16af:	45 31 c0             	xor    %r8d,%r8d
    16b2:	31 c9                	xor    %ecx,%ecx
    16b4:	48 8d 3d 15 fa ff ff 	lea    -0x5eb(%rip),%rdi        # 10d0 <main>
    16bb:	ff 15 ff 28 00 00    	call   *0x28ff(%rip)        # 3fc0 <__libc_start_main@GLIBC_2.34>
    16c1:	f4                   	hlt
    16c2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    16c9:	00 00 00 
    16cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000016d0 <deregister_tm_clones>:
    16d0:	48 8d 3d 81 29 00 00 	lea    0x2981(%rip),%rdi        # 4058 <__TMC_END__>
    16d7:	48 8d 05 7a 29 00 00 	lea    0x297a(%rip),%rax        # 4058 <__TMC_END__>
    16de:	48 39 f8             	cmp    %rdi,%rax
    16e1:	74 15                	je     16f8 <deregister_tm_clones+0x28>
    16e3:	48 8b 05 de 28 00 00 	mov    0x28de(%rip),%rax        # 3fc8 <_ITM_deregisterTMCloneTable@Base>
    16ea:	48 85 c0             	test   %rax,%rax
    16ed:	74 09                	je     16f8 <deregister_tm_clones+0x28>
    16ef:	ff e0                	jmp    *%rax
    16f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    16f8:	c3                   	ret
    16f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001700 <register_tm_clones>:
    1700:	48 8d 3d 51 29 00 00 	lea    0x2951(%rip),%rdi        # 4058 <__TMC_END__>
    1707:	48 8d 35 4a 29 00 00 	lea    0x294a(%rip),%rsi        # 4058 <__TMC_END__>
    170e:	48 29 fe             	sub    %rdi,%rsi
    1711:	48 89 f0             	mov    %rsi,%rax
    1714:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1718:	48 c1 f8 03          	sar    $0x3,%rax
    171c:	48 01 c6             	add    %rax,%rsi
    171f:	48 d1 fe             	sar    %rsi
    1722:	74 14                	je     1738 <register_tm_clones+0x38>
    1724:	48 8b 05 ad 28 00 00 	mov    0x28ad(%rip),%rax        # 3fd8 <_ITM_registerTMCloneTable@Base>
    172b:	48 85 c0             	test   %rax,%rax
    172e:	74 08                	je     1738 <register_tm_clones+0x38>
    1730:	ff e0                	jmp    *%rax
    1732:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1738:	c3                   	ret
    1739:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001740 <__do_global_dtors_aux>:
    1740:	f3 0f 1e fa          	endbr64
    1744:	80 3d 1d 29 00 00 00 	cmpb   $0x0,0x291d(%rip)        # 4068 <completed.0>
    174b:	75 2b                	jne    1778 <__do_global_dtors_aux+0x38>
    174d:	55                   	push   %rbp
    174e:	48 83 3d 8a 28 00 00 	cmpq   $0x0,0x288a(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    1755:	00 
    1756:	48 89 e5             	mov    %rsp,%rbp
    1759:	74 0c                	je     1767 <__do_global_dtors_aux+0x27>
    175b:	48 8b 3d ee 28 00 00 	mov    0x28ee(%rip),%rdi        # 4050 <__dso_handle>
    1762:	e8 59 f9 ff ff       	call   10c0 <__cxa_finalize@plt>
    1767:	e8 64 ff ff ff       	call   16d0 <deregister_tm_clones>
    176c:	c6 05 f5 28 00 00 01 	movb   $0x1,0x28f5(%rip)        # 4068 <completed.0>
    1773:	5d                   	pop    %rbp
    1774:	c3                   	ret
    1775:	0f 1f 00             	nopl   (%rax)
    1778:	c3                   	ret
    1779:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001780 <frame_dummy>:
    1780:	f3 0f 1e fa          	endbr64
    1784:	e9 77 ff ff ff       	jmp    1700 <register_tm_clones>

Disassembly of section .fini:

000000000000178c <_fini>:
    178c:	48 83 ec 08          	sub    $0x8,%rsp
    1790:	48 83 c4 08          	add    $0x8,%rsp
    1794:	c3                   	ret
