
bin/seq_bench/gcc/matmul_O0:     file format elf64-x86-64


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

0000000000001070 <fprintf@plt>:
    1070:	ff 25 aa 2f 00 00    	jmp    *0x2faa(%rip)        # 4020 <fprintf@GLIBC_2.2.5>
    1076:	68 04 00 00 00       	push   $0x4
    107b:	e9 a0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001080 <malloc@plt>:
    1080:	ff 25 a2 2f 00 00    	jmp    *0x2fa2(%rip)        # 4028 <malloc@GLIBC_2.2.5>
    1086:	68 05 00 00 00       	push   $0x5
    108b:	e9 90 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001090 <fopen@plt>:
    1090:	ff 25 9a 2f 00 00    	jmp    *0x2f9a(%rip)        # 4030 <fopen@GLIBC_2.2.5>
    1096:	68 06 00 00 00       	push   $0x6
    109b:	e9 80 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010a0 <perror@plt>:
    10a0:	ff 25 92 2f 00 00    	jmp    *0x2f92(%rip)        # 4038 <perror@GLIBC_2.2.5>
    10a6:	68 07 00 00 00       	push   $0x7
    10ab:	e9 70 ff ff ff       	jmp    1020 <_init+0x20>

Disassembly of section .plt.got:

00000000000010b0 <__cxa_finalize@plt>:
    10b0:	ff 25 2a 2f 00 00    	jmp    *0x2f2a(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    10b6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000010c0 <_start>:
    10c0:	31 ed                	xor    %ebp,%ebp
    10c2:	49 89 d1             	mov    %rdx,%r9
    10c5:	5e                   	pop    %rsi
    10c6:	48 89 e2             	mov    %rsp,%rdx
    10c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    10cd:	50                   	push   %rax
    10ce:	54                   	push   %rsp
    10cf:	45 31 c0             	xor    %r8d,%r8d
    10d2:	31 c9                	xor    %ecx,%ecx
    10d4:	48 8d 3d ce 00 00 00 	lea    0xce(%rip),%rdi        # 11a9 <main>
    10db:	ff 15 df 2e 00 00    	call   *0x2edf(%rip)        # 3fc0 <__libc_start_main@GLIBC_2.34>
    10e1:	f4                   	hlt
    10e2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    10e9:	00 00 00 
    10ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000010f0 <deregister_tm_clones>:
    10f0:	48 8d 3d 59 2f 00 00 	lea    0x2f59(%rip),%rdi        # 4050 <__TMC_END__>
    10f7:	48 8d 05 52 2f 00 00 	lea    0x2f52(%rip),%rax        # 4050 <__TMC_END__>
    10fe:	48 39 f8             	cmp    %rdi,%rax
    1101:	74 15                	je     1118 <deregister_tm_clones+0x28>
    1103:	48 8b 05 be 2e 00 00 	mov    0x2ebe(%rip),%rax        # 3fc8 <_ITM_deregisterTMCloneTable@Base>
    110a:	48 85 c0             	test   %rax,%rax
    110d:	74 09                	je     1118 <deregister_tm_clones+0x28>
    110f:	ff e0                	jmp    *%rax
    1111:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1118:	c3                   	ret
    1119:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001120 <register_tm_clones>:
    1120:	48 8d 3d 29 2f 00 00 	lea    0x2f29(%rip),%rdi        # 4050 <__TMC_END__>
    1127:	48 8d 35 22 2f 00 00 	lea    0x2f22(%rip),%rsi        # 4050 <__TMC_END__>
    112e:	48 29 fe             	sub    %rdi,%rsi
    1131:	48 89 f0             	mov    %rsi,%rax
    1134:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1138:	48 c1 f8 03          	sar    $0x3,%rax
    113c:	48 01 c6             	add    %rax,%rsi
    113f:	48 d1 fe             	sar    %rsi
    1142:	74 14                	je     1158 <register_tm_clones+0x38>
    1144:	48 8b 05 8d 2e 00 00 	mov    0x2e8d(%rip),%rax        # 3fd8 <_ITM_registerTMCloneTable@Base>
    114b:	48 85 c0             	test   %rax,%rax
    114e:	74 08                	je     1158 <register_tm_clones+0x38>
    1150:	ff e0                	jmp    *%rax
    1152:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1158:	c3                   	ret
    1159:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001160 <__do_global_dtors_aux>:
    1160:	f3 0f 1e fa          	endbr64
    1164:	80 3d fd 2e 00 00 00 	cmpb   $0x0,0x2efd(%rip)        # 4068 <completed.0>
    116b:	75 2b                	jne    1198 <__do_global_dtors_aux+0x38>
    116d:	55                   	push   %rbp
    116e:	48 83 3d 6a 2e 00 00 	cmpq   $0x0,0x2e6a(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    1175:	00 
    1176:	48 89 e5             	mov    %rsp,%rbp
    1179:	74 0c                	je     1187 <__do_global_dtors_aux+0x27>
    117b:	48 8b 3d c6 2e 00 00 	mov    0x2ec6(%rip),%rdi        # 4048 <__dso_handle>
    1182:	e8 29 ff ff ff       	call   10b0 <__cxa_finalize@plt>
    1187:	e8 64 ff ff ff       	call   10f0 <deregister_tm_clones>
    118c:	c6 05 d5 2e 00 00 01 	movb   $0x1,0x2ed5(%rip)        # 4068 <completed.0>
    1193:	5d                   	pop    %rbp
    1194:	c3                   	ret
    1195:	0f 1f 00             	nopl   (%rax)
    1198:	c3                   	ret
    1199:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000011a0 <frame_dummy>:
    11a0:	f3 0f 1e fa          	endbr64
    11a4:	e9 77 ff ff ff       	jmp    1120 <register_tm_clones>

00000000000011a9 <main>:
    11a9:	55                   	push   %rbp
    11aa:	48 89 e5             	mov    %rsp,%rbp
    11ad:	48 83 ec 60          	sub    $0x60,%rsp
    11b1:	89 7d ac             	mov    %edi,-0x54(%rbp)
    11b4:	48 89 75 a0          	mov    %rsi,-0x60(%rbp)
    11b8:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    11bd:	e8 be fe ff ff       	call   1080 <malloc@plt>
    11c2:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    11c6:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    11cb:	e8 b0 fe ff ff       	call   1080 <malloc@plt>
    11d0:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    11d4:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    11d9:	e8 a2 fe ff ff       	call   1080 <malloc@plt>
    11de:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    11e2:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
    11e9:	e9 85 00 00 00       	jmp    1273 <main+0xca>
    11ee:	c7 45 f8 00 00 00 00 	movl   $0x0,-0x8(%rbp)
    11f5:	eb 6f                	jmp    1266 <main+0xbd>
    11f7:	8b 45 fc             	mov    -0x4(%rbp),%eax
    11fa:	48 98                	cltq
    11fc:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    1203:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    1207:	48 01 c2             	add    %rax,%rdx
    120a:	8b 45 f8             	mov    -0x8(%rbp),%eax
    120d:	48 98                	cltq
    120f:	f2 0f 10 05 31 0e 00 	movsd  0xe31(%rip),%xmm0        # 2048 <_IO_stdin_used+0x48>
    1216:	00 
    1217:	f2 0f 11 04 c2       	movsd  %xmm0,(%rdx,%rax,8)
    121c:	8b 45 fc             	mov    -0x4(%rbp),%eax
    121f:	48 98                	cltq
    1221:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    1228:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    122c:	48 01 c2             	add    %rax,%rdx
    122f:	8b 45 f8             	mov    -0x8(%rbp),%eax
    1232:	48 98                	cltq
    1234:	f2 0f 10 05 14 0e 00 	movsd  0xe14(%rip),%xmm0        # 2050 <_IO_stdin_used+0x50>
    123b:	00 
    123c:	f2 0f 11 04 c2       	movsd  %xmm0,(%rdx,%rax,8)
    1241:	8b 45 fc             	mov    -0x4(%rbp),%eax
    1244:	48 98                	cltq
    1246:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    124d:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    1251:	48 01 c2             	add    %rax,%rdx
    1254:	8b 45 f8             	mov    -0x8(%rbp),%eax
    1257:	48 98                	cltq
    1259:	66 0f ef c0          	pxor   %xmm0,%xmm0
    125d:	f2 0f 11 04 c2       	movsd  %xmm0,(%rdx,%rax,8)
    1262:	83 45 f8 01          	addl   $0x1,-0x8(%rbp)
    1266:	81 7d f8 87 13 00 00 	cmpl   $0x1387,-0x8(%rbp)
    126d:	7e 88                	jle    11f7 <main+0x4e>
    126f:	83 45 fc 01          	addl   $0x1,-0x4(%rbp)
    1273:	81 7d fc 87 13 00 00 	cmpl   $0x1387,-0x4(%rbp)
    127a:	0f 8e 6e ff ff ff    	jle    11ee <main+0x45>
    1280:	e8 bb fd ff ff       	call   1040 <clock@plt>
    1285:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    1289:	c7 45 f4 00 00 00 00 	movl   $0x0,-0xc(%rbp)
    1290:	e9 ba 00 00 00       	jmp    134f <main+0x1a6>
    1295:	c7 45 f0 00 00 00 00 	movl   $0x0,-0x10(%rbp)
    129c:	e9 9d 00 00 00       	jmp    133e <main+0x195>
    12a1:	c7 45 ec 00 00 00 00 	movl   $0x0,-0x14(%rbp)
    12a8:	e9 80 00 00 00       	jmp    132d <main+0x184>
    12ad:	8b 45 f4             	mov    -0xc(%rbp),%eax
    12b0:	48 98                	cltq
    12b2:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    12b9:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    12bd:	48 01 c2             	add    %rax,%rdx
    12c0:	8b 45 ec             	mov    -0x14(%rbp),%eax
    12c3:	48 98                	cltq
    12c5:	f2 0f 10 0c c2       	movsd  (%rdx,%rax,8),%xmm1
    12ca:	8b 45 f4             	mov    -0xc(%rbp),%eax
    12cd:	48 98                	cltq
    12cf:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    12d6:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    12da:	48 01 c2             	add    %rax,%rdx
    12dd:	8b 45 f0             	mov    -0x10(%rbp),%eax
    12e0:	48 98                	cltq
    12e2:	f2 0f 10 14 c2       	movsd  (%rdx,%rax,8),%xmm2
    12e7:	8b 45 f0             	mov    -0x10(%rbp),%eax
    12ea:	48 98                	cltq
    12ec:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    12f3:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    12f7:	48 01 c2             	add    %rax,%rdx
    12fa:	8b 45 ec             	mov    -0x14(%rbp),%eax
    12fd:	48 98                	cltq
    12ff:	f2 0f 10 04 c2       	movsd  (%rdx,%rax,8),%xmm0
    1304:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
    1308:	8b 45 f4             	mov    -0xc(%rbp),%eax
    130b:	48 98                	cltq
    130d:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    1314:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    1318:	48 01 c2             	add    %rax,%rdx
    131b:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    131f:	8b 45 ec             	mov    -0x14(%rbp),%eax
    1322:	48 98                	cltq
    1324:	f2 0f 11 04 c2       	movsd  %xmm0,(%rdx,%rax,8)
    1329:	83 45 ec 01          	addl   $0x1,-0x14(%rbp)
    132d:	81 7d ec 87 13 00 00 	cmpl   $0x1387,-0x14(%rbp)
    1334:	0f 8e 73 ff ff ff    	jle    12ad <main+0x104>
    133a:	83 45 f0 01          	addl   $0x1,-0x10(%rbp)
    133e:	81 7d f0 87 13 00 00 	cmpl   $0x1387,-0x10(%rbp)
    1345:	0f 8e 56 ff ff ff    	jle    12a1 <main+0xf8>
    134b:	83 45 f4 01          	addl   $0x1,-0xc(%rbp)
    134f:	81 7d f4 87 13 00 00 	cmpl   $0x1387,-0xc(%rbp)
    1356:	0f 8e 39 ff ff ff    	jle    1295 <main+0xec>
    135c:	e8 df fc ff ff       	call   1040 <clock@plt>
    1361:	48 2b 45 c0          	sub    -0x40(%rbp),%rax
    1365:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    1369:	66 0f ef c0          	pxor   %xmm0,%xmm0
    136d:	f2 48 0f 2a 45 c0    	cvtsi2sdq -0x40(%rbp),%xmm0
    1373:	f2 0f 10 0d dd 0c 00 	movsd  0xcdd(%rip),%xmm1        # 2058 <_IO_stdin_used+0x58>
    137a:	00 
    137b:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    137f:	f2 0f 11 45 b8       	movsd  %xmm0,-0x48(%rbp)
    1384:	48 8b 05 d5 2c 00 00 	mov    0x2cd5(%rip),%rax        # 4060 <stderr@GLIBC_2.2.5>
    138b:	48 8b 55 b8          	mov    -0x48(%rbp),%rdx
    138f:	66 48 0f 6e c2       	movq   %rdx,%xmm0
    1394:	ba 88 13 00 00       	mov    $0x1388,%edx
    1399:	48 8d 0d 68 0c 00 00 	lea    0xc68(%rip),%rcx        # 2008 <_IO_stdin_used+0x8>
    13a0:	48 89 ce             	mov    %rcx,%rsi
    13a3:	48 89 c7             	mov    %rax,%rdi
    13a6:	b8 01 00 00 00       	mov    $0x1,%eax
    13ab:	e8 c0 fc ff ff       	call   1070 <fprintf@plt>
    13b0:	48 8d 05 6c 0c 00 00 	lea    0xc6c(%rip),%rax        # 2023 <_IO_stdin_used+0x23>
    13b7:	48 89 c6             	mov    %rax,%rsi
    13ba:	48 8d 05 64 0c 00 00 	lea    0xc64(%rip),%rax        # 2025 <_IO_stdin_used+0x25>
    13c1:	48 89 c7             	mov    %rax,%rdi
    13c4:	e8 c7 fc ff ff       	call   1090 <fopen@plt>
    13c9:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    13cd:	48 83 7d b0 00       	cmpq   $0x0,-0x50(%rbp)
    13d2:	75 19                	jne    13ed <main+0x244>
    13d4:	48 8d 05 56 0c 00 00 	lea    0xc56(%rip),%rax        # 2031 <_IO_stdin_used+0x31>
    13db:	48 89 c7             	mov    %rax,%rdi
    13de:	e8 bd fc ff ff       	call   10a0 <perror@plt>
    13e3:	b8 01 00 00 00       	mov    $0x1,%eax
    13e8:	e9 ce 00 00 00       	jmp    14bb <main+0x312>
    13ed:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    13f1:	ba 88 13 00 00       	mov    $0x1388,%edx
    13f6:	48 8d 0d 3a 0c 00 00 	lea    0xc3a(%rip),%rcx        # 2037 <_IO_stdin_used+0x37>
    13fd:	48 89 ce             	mov    %rcx,%rsi
    1400:	48 89 c7             	mov    %rax,%rdi
    1403:	b8 00 00 00 00       	mov    $0x0,%eax
    1408:	e8 63 fc ff ff       	call   1070 <fprintf@plt>
    140d:	c7 45 e8 00 00 00 00 	movl   $0x0,-0x18(%rbp)
    1414:	eb 67                	jmp    147d <main+0x2d4>
    1416:	c7 45 e4 00 00 00 00 	movl   $0x0,-0x1c(%rbp)
    141d:	eb 40                	jmp    145f <main+0x2b6>
    141f:	8b 45 e8             	mov    -0x18(%rbp),%eax
    1422:	48 98                	cltq
    1424:	48 69 d0 40 9c 00 00 	imul   $0x9c40,%rax,%rdx
    142b:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    142f:	48 01 c2             	add    %rax,%rdx
    1432:	8b 45 e4             	mov    -0x1c(%rbp),%eax
    1435:	48 98                	cltq
    1437:	48 8b 14 c2          	mov    (%rdx,%rax,8),%rdx
    143b:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    143f:	66 48 0f 6e c2       	movq   %rdx,%xmm0
    1444:	48 8d 15 f1 0b 00 00 	lea    0xbf1(%rip),%rdx        # 203c <_IO_stdin_used+0x3c>
    144b:	48 89 d6             	mov    %rdx,%rsi
    144e:	48 89 c7             	mov    %rax,%rdi
    1451:	b8 01 00 00 00       	mov    $0x1,%eax
    1456:	e8 15 fc ff ff       	call   1070 <fprintf@plt>
    145b:	83 45 e4 01          	addl   $0x1,-0x1c(%rbp)
    145f:	81 7d e4 e7 03 00 00 	cmpl   $0x3e7,-0x1c(%rbp)
    1466:	7e b7                	jle    141f <main+0x276>
    1468:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    146c:	48 89 c6             	mov    %rax,%rsi
    146f:	bf 0a 00 00 00       	mov    $0xa,%edi
    1474:	e8 e7 fb ff ff       	call   1060 <fputc@plt>
    1479:	83 45 e8 01          	addl   $0x1,-0x18(%rbp)
    147d:	81 7d e8 e7 03 00 00 	cmpl   $0x3e7,-0x18(%rbp)
    1484:	7e 90                	jle    1416 <main+0x26d>
    1486:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    148a:	48 89 c7             	mov    %rax,%rdi
    148d:	e8 be fb ff ff       	call   1050 <fclose@plt>
    1492:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    1496:	48 89 c7             	mov    %rax,%rdi
    1499:	e8 92 fb ff ff       	call   1030 <free@plt>
    149e:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    14a2:	48 89 c7             	mov    %rax,%rdi
    14a5:	e8 86 fb ff ff       	call   1030 <free@plt>
    14aa:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    14ae:	48 89 c7             	mov    %rax,%rdi
    14b1:	e8 7a fb ff ff       	call   1030 <free@plt>
    14b6:	b8 00 00 00 00       	mov    $0x0,%eax
    14bb:	c9                   	leave
    14bc:	c3                   	ret

Disassembly of section .fini:

00000000000014c0 <_fini>:
    14c0:	48 83 ec 08          	sub    $0x8,%rsp
    14c4:	48 83 c4 08          	add    $0x8,%rsp
    14c8:	c3                   	ret
