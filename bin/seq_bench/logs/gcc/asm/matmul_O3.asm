
bin/seq_bench/gcc/matmul_O3:     file format elf64-x86-64


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
    10d0:	41 57                	push   %r15
    10d2:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    10d7:	41 56                	push   %r14
    10d9:	41 55                	push   %r13
    10db:	41 54                	push   %r12
    10dd:	55                   	push   %rbp
    10de:	53                   	push   %rbx
    10df:	48 83 ec 18          	sub    $0x18,%rsp
    10e3:	e8 a8 ff ff ff       	call   1090 <malloc@plt>
    10e8:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    10ed:	49 89 c6             	mov    %rax,%r14
    10f0:	e8 9b ff ff ff       	call   1090 <malloc@plt>
    10f5:	be 01 00 00 00       	mov    $0x1,%esi
    10fa:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    10ff:	49 89 c5             	mov    %rax,%r13
    1102:	e8 69 ff ff ff       	call   1070 <calloc@plt>
    1107:	f2 0f 10 05 31 0f 00 	movsd  0xf31(%rip),%xmm0        # 2040 <_IO_stdin_used+0x40>
    110e:	00 
    110f:	49 8d 96 40 9c 00 00 	lea    0x9c40(%r14),%rdx
    1116:	4c 89 f1             	mov    %r14,%rcx
    1119:	48 89 04 24          	mov    %rax,(%rsp)
    111d:	49 8d b6 40 5e ec 0b 	lea    0xbec5e40(%r14),%rsi
    1124:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
    1128:	48 89 c8             	mov    %rcx,%rax
    112b:	0f 11 00             	movups %xmm0,(%rax)
    112e:	48 83 c0 10          	add    $0x10,%rax
    1132:	48 39 d0             	cmp    %rdx,%rax
    1135:	75 f4                	jne    112b <main+0x5b>
    1137:	48 8d 90 40 9c 00 00 	lea    0x9c40(%rax),%rdx
    113e:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    1145:	48 39 f2             	cmp    %rsi,%rdx
    1148:	75 de                	jne    1128 <main+0x58>
    114a:	f2 0f 10 05 f6 0e 00 	movsd  0xef6(%rip),%xmm0        # 2048 <_IO_stdin_used+0x48>
    1151:	00 
    1152:	49 8d 95 40 9c 00 00 	lea    0x9c40(%r13),%rdx
    1159:	49 8d b5 40 5e ec 0b 	lea    0xbec5e40(%r13),%rsi
    1160:	4c 89 e9             	mov    %r13,%rcx
    1163:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
    1167:	48 89 c8             	mov    %rcx,%rax
    116a:	0f 11 00             	movups %xmm0,(%rax)
    116d:	48 83 c0 10          	add    $0x10,%rax
    1171:	48 39 c2             	cmp    %rax,%rdx
    1174:	75 f4                	jne    116a <main+0x9a>
    1176:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
    117d:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    1184:	48 39 f2             	cmp    %rsi,%rdx
    1187:	75 de                	jne    1167 <main+0x97>
    1189:	e8 b2 fe ff ff       	call   1040 <clock@plt>
    118e:	45 31 c9             	xor    %r9d,%r9d
    1191:	4d 8d 5e 08          	lea    0x8(%r14),%r11
    1195:	48 89 c3             	mov    %rax,%rbx
    1198:	48 8b 04 24          	mov    (%rsp),%rax
    119c:	4b 8d 3c 0b          	lea    (%r11,%r9,1),%rdi
    11a0:	4c 89 e9             	mov    %r13,%rcx
    11a3:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    11a9:	4e 8d 14 08          	lea    (%rax,%r9,1),%r10
    11ad:	f2 0f 10 67 f8       	movsd  -0x8(%rdi),%xmm4
    11b2:	f2 0f 10 1f          	movsd  (%rdi),%xmm3
    11b6:	4c 89 d2             	mov    %r10,%rdx
    11b9:	31 c0                	xor    %eax,%eax
    11bb:	48 8d b1 40 9c 00 00 	lea    0x9c40(%rcx),%rsi
    11c2:	66 0f 14 e4          	unpcklpd %xmm4,%xmm4
    11c6:	66 0f 14 db          	unpcklpd %xmm3,%xmm3
    11ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    11d0:	66 0f 10 44 01 10    	movupd 0x10(%rcx,%rax,1),%xmm0
    11d6:	66 0f 10 4c 06 10    	movupd 0x10(%rsi,%rax,1),%xmm1
    11dc:	48 83 c2 20          	add    $0x20,%rdx
    11e0:	66 0f 10 6a f0       	movupd -0x10(%rdx),%xmm5
    11e5:	66 0f 10 14 06       	movupd (%rsi,%rax,1),%xmm2
    11ea:	66 0f 59 c4          	mulpd  %xmm4,%xmm0
    11ee:	66 0f 10 72 e0       	movupd -0x20(%rdx),%xmm6
    11f3:	66 0f 59 cb          	mulpd  %xmm3,%xmm1
    11f7:	66 0f 59 d3          	mulpd  %xmm3,%xmm2
    11fb:	66 0f 58 c5          	addpd  %xmm5,%xmm0
    11ff:	66 0f 58 c1          	addpd  %xmm1,%xmm0
    1203:	66 0f 10 0c 01       	movupd (%rcx,%rax,1),%xmm1
    1208:	48 83 c0 20          	add    $0x20,%rax
    120c:	66 0f 59 cc          	mulpd  %xmm4,%xmm1
    1210:	0f 11 42 f0          	movups %xmm0,-0x10(%rdx)
    1214:	66 0f 58 ce          	addpd  %xmm6,%xmm1
    1218:	66 0f 58 ca          	addpd  %xmm2,%xmm1
    121c:	0f 11 4a e0          	movups %xmm1,-0x20(%rdx)
    1220:	48 3d 40 9c 00 00    	cmp    $0x9c40,%rax
    1226:	75 a8                	jne    11d0 <main+0x100>
    1228:	41 83 c0 02          	add    $0x2,%r8d
    122c:	48 81 c1 80 38 01 00 	add    $0x13880,%rcx
    1233:	48 83 c7 10          	add    $0x10,%rdi
    1237:	41 81 f8 89 13 00 00 	cmp    $0x1389,%r8d
    123e:	0f 85 69 ff ff ff    	jne    11ad <main+0xdd>
    1244:	49 81 c1 40 9c 00 00 	add    $0x9c40,%r9
    124b:	49 81 f9 00 c2 eb 0b 	cmp    $0xbebc200,%r9
    1252:	0f 85 40 ff ff ff    	jne    1198 <main+0xc8>
    1258:	e8 e3 fd ff ff       	call   1040 <clock@plt>
    125d:	48 8b 3d fc 2d 00 00 	mov    0x2dfc(%rip),%rdi        # 4060 <stderr@GLIBC_2.2.5>
    1264:	66 0f ef c0          	pxor   %xmm0,%xmm0
    1268:	ba 88 13 00 00       	mov    $0x1388,%edx
    126d:	48 29 d8             	sub    %rbx,%rax
    1270:	48 8d 35 8d 0d 00 00 	lea    0xd8d(%rip),%rsi        # 2004 <_IO_stdin_used+0x4>
    1277:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    127c:	b8 01 00 00 00       	mov    $0x1,%eax
    1281:	f2 0f 5e 05 c7 0d 00 	divsd  0xdc7(%rip),%xmm0        # 2050 <_IO_stdin_used+0x50>
    1288:	00 
    1289:	e8 f2 fd ff ff       	call   1080 <fprintf@plt>
    128e:	48 8d 35 8a 0d 00 00 	lea    0xd8a(%rip),%rsi        # 201f <_IO_stdin_used+0x1f>
    1295:	48 8d 3d 85 0d 00 00 	lea    0xd85(%rip),%rdi        # 2021 <_IO_stdin_used+0x21>
    129c:	e8 ff fd ff ff       	call   10a0 <fopen@plt>
    12a1:	48 89 c3             	mov    %rax,%rbx
    12a4:	48 85 c0             	test   %rax,%rax
    12a7:	0f 84 a8 00 00 00    	je     1355 <main+0x285>
    12ad:	48 89 c7             	mov    %rax,%rdi
    12b0:	ba 88 13 00 00       	mov    $0x1388,%edx
    12b5:	31 c0                	xor    %eax,%eax
    12b7:	48 8d 35 75 0d 00 00 	lea    0xd75(%rip),%rsi        # 2033 <_IO_stdin_used+0x33>
    12be:	4c 8d 25 73 0d 00 00 	lea    0xd73(%rip),%r12        # 2038 <_IO_stdin_used+0x38>
    12c5:	e8 b6 fd ff ff       	call   1080 <fprintf@plt>
    12ca:	48 8b 04 24          	mov    (%rsp),%rax
    12ce:	48 8d a8 40 1f 00 00 	lea    0x1f40(%rax),%rbp
    12d5:	48 05 40 79 62 02    	add    $0x2627940,%rax
    12db:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    12e0:	4c 8d bd c0 e0 ff ff 	lea    -0x1f40(%rbp),%r15
    12e7:	f2 41 0f 10 07       	movsd  (%r15),%xmm0
    12ec:	4c 89 e6             	mov    %r12,%rsi
    12ef:	48 89 df             	mov    %rbx,%rdi
    12f2:	b8 01 00 00 00       	mov    $0x1,%eax
    12f7:	49 83 c7 08          	add    $0x8,%r15
    12fb:	e8 80 fd ff ff       	call   1080 <fprintf@plt>
    1300:	49 39 ef             	cmp    %rbp,%r15
    1303:	75 e2                	jne    12e7 <main+0x217>
    1305:	48 89 de             	mov    %rbx,%rsi
    1308:	bf 0a 00 00 00       	mov    $0xa,%edi
    130d:	49 8d af 40 9c 00 00 	lea    0x9c40(%r15),%rbp
    1314:	e8 47 fd ff ff       	call   1060 <fputc@plt>
    1319:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    131e:	48 39 c5             	cmp    %rax,%rbp
    1321:	75 bd                	jne    12e0 <main+0x210>
    1323:	48 89 df             	mov    %rbx,%rdi
    1326:	e8 25 fd ff ff       	call   1050 <fclose@plt>
    132b:	4c 89 f7             	mov    %r14,%rdi
    132e:	e8 fd fc ff ff       	call   1030 <free@plt>
    1333:	4c 89 ef             	mov    %r13,%rdi
    1336:	e8 f5 fc ff ff       	call   1030 <free@plt>
    133b:	48 8b 3c 24          	mov    (%rsp),%rdi
    133f:	e8 ec fc ff ff       	call   1030 <free@plt>
    1344:	31 c0                	xor    %eax,%eax
    1346:	48 83 c4 18          	add    $0x18,%rsp
    134a:	5b                   	pop    %rbx
    134b:	5d                   	pop    %rbp
    134c:	41 5c                	pop    %r12
    134e:	41 5d                	pop    %r13
    1350:	41 5e                	pop    %r14
    1352:	41 5f                	pop    %r15
    1354:	c3                   	ret
    1355:	48 8d 3d d1 0c 00 00 	lea    0xcd1(%rip),%rdi        # 202d <_IO_stdin_used+0x2d>
    135c:	e8 4f fd ff ff       	call   10b0 <perror@plt>
    1361:	b8 01 00 00 00       	mov    $0x1,%eax
    1366:	eb de                	jmp    1346 <main+0x276>
    1368:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    136f:	00 

0000000000001370 <_start>:
    1370:	31 ed                	xor    %ebp,%ebp
    1372:	49 89 d1             	mov    %rdx,%r9
    1375:	5e                   	pop    %rsi
    1376:	48 89 e2             	mov    %rsp,%rdx
    1379:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    137d:	50                   	push   %rax
    137e:	54                   	push   %rsp
    137f:	45 31 c0             	xor    %r8d,%r8d
    1382:	31 c9                	xor    %ecx,%ecx
    1384:	48 8d 3d 45 fd ff ff 	lea    -0x2bb(%rip),%rdi        # 10d0 <main>
    138b:	ff 15 2f 2c 00 00    	call   *0x2c2f(%rip)        # 3fc0 <__libc_start_main@GLIBC_2.34>
    1391:	f4                   	hlt
    1392:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    1399:	00 00 00 
    139c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000013a0 <deregister_tm_clones>:
    13a0:	48 8d 3d b1 2c 00 00 	lea    0x2cb1(%rip),%rdi        # 4058 <__TMC_END__>
    13a7:	48 8d 05 aa 2c 00 00 	lea    0x2caa(%rip),%rax        # 4058 <__TMC_END__>
    13ae:	48 39 f8             	cmp    %rdi,%rax
    13b1:	74 15                	je     13c8 <deregister_tm_clones+0x28>
    13b3:	48 8b 05 0e 2c 00 00 	mov    0x2c0e(%rip),%rax        # 3fc8 <_ITM_deregisterTMCloneTable@Base>
    13ba:	48 85 c0             	test   %rax,%rax
    13bd:	74 09                	je     13c8 <deregister_tm_clones+0x28>
    13bf:	ff e0                	jmp    *%rax
    13c1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    13c8:	c3                   	ret
    13c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000013d0 <register_tm_clones>:
    13d0:	48 8d 3d 81 2c 00 00 	lea    0x2c81(%rip),%rdi        # 4058 <__TMC_END__>
    13d7:	48 8d 35 7a 2c 00 00 	lea    0x2c7a(%rip),%rsi        # 4058 <__TMC_END__>
    13de:	48 29 fe             	sub    %rdi,%rsi
    13e1:	48 89 f0             	mov    %rsi,%rax
    13e4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    13e8:	48 c1 f8 03          	sar    $0x3,%rax
    13ec:	48 01 c6             	add    %rax,%rsi
    13ef:	48 d1 fe             	sar    %rsi
    13f2:	74 14                	je     1408 <register_tm_clones+0x38>
    13f4:	48 8b 05 dd 2b 00 00 	mov    0x2bdd(%rip),%rax        # 3fd8 <_ITM_registerTMCloneTable@Base>
    13fb:	48 85 c0             	test   %rax,%rax
    13fe:	74 08                	je     1408 <register_tm_clones+0x38>
    1400:	ff e0                	jmp    *%rax
    1402:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1408:	c3                   	ret
    1409:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001410 <__do_global_dtors_aux>:
    1410:	f3 0f 1e fa          	endbr64
    1414:	80 3d 4d 2c 00 00 00 	cmpb   $0x0,0x2c4d(%rip)        # 4068 <completed.0>
    141b:	75 2b                	jne    1448 <__do_global_dtors_aux+0x38>
    141d:	55                   	push   %rbp
    141e:	48 83 3d ba 2b 00 00 	cmpq   $0x0,0x2bba(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    1425:	00 
    1426:	48 89 e5             	mov    %rsp,%rbp
    1429:	74 0c                	je     1437 <__do_global_dtors_aux+0x27>
    142b:	48 8b 3d 1e 2c 00 00 	mov    0x2c1e(%rip),%rdi        # 4050 <__dso_handle>
    1432:	e8 89 fc ff ff       	call   10c0 <__cxa_finalize@plt>
    1437:	e8 64 ff ff ff       	call   13a0 <deregister_tm_clones>
    143c:	c6 05 25 2c 00 00 01 	movb   $0x1,0x2c25(%rip)        # 4068 <completed.0>
    1443:	5d                   	pop    %rbp
    1444:	c3                   	ret
    1445:	0f 1f 00             	nopl   (%rax)
    1448:	c3                   	ret
    1449:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001450 <frame_dummy>:
    1450:	f3 0f 1e fa          	endbr64
    1454:	e9 77 ff ff ff       	jmp    13d0 <register_tm_clones>

Disassembly of section .fini:

000000000000145c <_fini>:
    145c:	48 83 ec 08          	sub    $0x8,%rsp
    1460:	48 83 c4 08          	add    $0x8,%rsp
    1464:	c3                   	ret
