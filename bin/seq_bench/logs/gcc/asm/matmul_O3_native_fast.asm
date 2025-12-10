
bin/seq_bench/gcc/matmul_O3_native_fast:     file format elf64-x86-64


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
    1126:	49 8d 97 40 9c 00 00 	lea    0x9c40(%r15),%rdx
    112d:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    1131:	49 8d b7 40 5e ec 0b 	lea    0xbec5e40(%r15),%rsi
    1138:	48 89 c8             	mov    %rcx,%rax
    113b:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    113f:	48 83 c0 20          	add    $0x20,%rax
    1143:	48 39 d0             	cmp    %rdx,%rax
    1146:	75 f3                	jne    113b <main+0x6b>
    1148:	48 8d 90 40 9c 00 00 	lea    0x9c40(%rax),%rdx
    114f:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    1156:	48 39 f2             	cmp    %rsi,%rdx
    1159:	75 dd                	jne    1138 <main+0x68>
    115b:	c4 e2 7d 19 05 e4 0e 	vbroadcastsd 0xee4(%rip),%ymm0        # 2048 <_IO_stdin_used+0x48>
    1162:	00 00 
    1164:	48 8b 4d c8          	mov    -0x38(%rbp),%rcx
    1168:	48 8d 91 40 9c 00 00 	lea    0x9c40(%rcx),%rdx
    116f:	48 8d b1 40 5e ec 0b 	lea    0xbec5e40(%rcx),%rsi
    1176:	48 89 c8             	mov    %rcx,%rax
    1179:	c5 fd 11 00          	vmovupd %ymm0,(%rax)
    117d:	48 83 c0 20          	add    $0x20,%rax
    1181:	48 39 c2             	cmp    %rax,%rdx
    1184:	75 f3                	jne    1179 <main+0xa9>
    1186:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
    118d:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    1194:	48 39 f2             	cmp    %rsi,%rdx
    1197:	75 dd                	jne    1176 <main+0xa6>
    1199:	c5 f8 77             	vzeroupper
    119c:	e8 9f fe ff ff       	call   1040 <clock@plt>
    11a1:	4c 8b 4d c0          	mov    -0x40(%rbp),%r9
    11a5:	4d 8d 57 08          	lea    0x8(%r15),%r10
    11a9:	48 89 c3             	mov    %rax,%rbx
    11ac:	4d 8d 99 00 c2 eb 0b 	lea    0xbebc200(%r9),%r11
    11b3:	48 8b 4d c8          	mov    -0x38(%rbp),%rcx
    11b7:	4c 89 d7             	mov    %r10,%rdi
    11ba:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    11c0:	c4 e2 7d 19 5f f8    	vbroadcastsd -0x8(%rdi),%ymm3
    11c6:	c4 e2 7d 19 17       	vbroadcastsd (%rdi),%ymm2
    11cb:	4c 89 ca             	mov    %r9,%rdx
    11ce:	31 c0                	xor    %eax,%eax
    11d0:	48 8d b1 40 9c 00 00 	lea    0x9c40(%rcx),%rsi
    11d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    11de:	00 00 
    11e0:	c5 fd 10 44 08 20    	vmovupd 0x20(%rax,%rcx,1),%ymm0
    11e6:	c5 fd 10 0c 01       	vmovupd (%rcx,%rax,1),%ymm1
    11eb:	48 83 c2 40          	add    $0x40,%rdx
    11ef:	c4 e2 e5 a8 4a c0    	vfmadd213pd -0x40(%rdx),%ymm3,%ymm1
    11f5:	c4 e2 e5 a8 42 e0    	vfmadd213pd -0x20(%rdx),%ymm3,%ymm0
    11fb:	c4 e2 ed b8 0c 06    	vfmadd231pd (%rsi,%rax,1),%ymm2,%ymm1
    1201:	c4 e2 ed b8 44 06 20 	vfmadd231pd 0x20(%rsi,%rax,1),%ymm2,%ymm0
    1208:	48 83 c0 40          	add    $0x40,%rax
    120c:	c5 fd 11 4a c0       	vmovupd %ymm1,-0x40(%rdx)
    1211:	c5 fd 11 42 e0       	vmovupd %ymm0,-0x20(%rdx)
    1216:	48 3d 40 9c 00 00    	cmp    $0x9c40,%rax
    121c:	75 c2                	jne    11e0 <main+0x110>
    121e:	41 83 c0 02          	add    $0x2,%r8d
    1222:	48 81 c1 80 38 01 00 	add    $0x13880,%rcx
    1229:	48 83 c7 10          	add    $0x10,%rdi
    122d:	41 81 f8 89 13 00 00 	cmp    $0x1389,%r8d
    1234:	75 8a                	jne    11c0 <main+0xf0>
    1236:	49 81 c1 40 9c 00 00 	add    $0x9c40,%r9
    123d:	49 81 c2 40 9c 00 00 	add    $0x9c40,%r10
    1244:	4d 39 cb             	cmp    %r9,%r11
    1247:	0f 85 66 ff ff ff    	jne    11b3 <main+0xe3>
    124d:	c5 f8 77             	vzeroupper
    1250:	e8 eb fd ff ff       	call   1040 <clock@plt>
    1255:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    1259:	48 8b 3d 00 2e 00 00 	mov    0x2e00(%rip),%rdi        # 4060 <stderr@GLIBC_2.2.5>
    1260:	48 29 d8             	sub    %rbx,%rax
    1263:	ba 88 13 00 00       	mov    $0x1388,%edx
    1268:	48 8d 35 95 0d 00 00 	lea    0xd95(%rip),%rsi        # 2004 <_IO_stdin_used+0x4>
    126f:	c4 e1 fb 2a c0       	vcvtsi2sd %rax,%xmm0,%xmm0
    1274:	c5 fb 59 05 d4 0d 00 	vmulsd 0xdd4(%rip),%xmm0,%xmm0        # 2050 <_IO_stdin_used+0x50>
    127b:	00 
    127c:	b8 01 00 00 00       	mov    $0x1,%eax
    1281:	e8 fa fd ff ff       	call   1080 <fprintf@plt>
    1286:	48 8d 35 92 0d 00 00 	lea    0xd92(%rip),%rsi        # 201f <_IO_stdin_used+0x1f>
    128d:	48 8d 3d 8d 0d 00 00 	lea    0xd8d(%rip),%rdi        # 2021 <_IO_stdin_used+0x21>
    1294:	e8 07 fe ff ff       	call   10a0 <fopen@plt>
    1299:	48 85 c0             	test   %rax,%rax
    129c:	48 89 c3             	mov    %rax,%rbx
    129f:	0f 84 ae 00 00 00    	je     1353 <main+0x283>
    12a5:	48 89 c7             	mov    %rax,%rdi
    12a8:	ba 88 13 00 00       	mov    $0x1388,%edx
    12ad:	31 c0                	xor    %eax,%eax
    12af:	48 8d 35 7d 0d 00 00 	lea    0xd7d(%rip),%rsi        # 2033 <_IO_stdin_used+0x33>
    12b6:	4c 8d 2d 7b 0d 00 00 	lea    0xd7b(%rip),%r13        # 2038 <_IO_stdin_used+0x38>
    12bd:	e8 be fd ff ff       	call   1080 <fprintf@plt>
    12c2:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
    12c6:	4c 8d a0 40 1f 00 00 	lea    0x1f40(%rax),%r12
    12cd:	48 05 40 79 62 02    	add    $0x2627940,%rax
    12d3:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
    12d7:	4d 8d b4 24 c0 e0 ff 	lea    -0x1f40(%r12),%r14
    12de:	ff 
    12df:	c4 c1 7b 10 06       	vmovsd (%r14),%xmm0
    12e4:	4c 89 ee             	mov    %r13,%rsi
    12e7:	48 89 df             	mov    %rbx,%rdi
    12ea:	b8 01 00 00 00       	mov    $0x1,%eax
    12ef:	49 83 c6 08          	add    $0x8,%r14
    12f3:	e8 88 fd ff ff       	call   1080 <fprintf@plt>
    12f8:	4d 39 e6             	cmp    %r12,%r14
    12fb:	75 e2                	jne    12df <main+0x20f>
    12fd:	48 89 de             	mov    %rbx,%rsi
    1300:	bf 0a 00 00 00       	mov    $0xa,%edi
    1305:	4d 8d a6 40 9c 00 00 	lea    0x9c40(%r14),%r12
    130c:	e8 4f fd ff ff       	call   1060 <fputc@plt>
    1311:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
    1315:	49 39 c4             	cmp    %rax,%r12
    1318:	75 bd                	jne    12d7 <main+0x207>
    131a:	48 89 df             	mov    %rbx,%rdi
    131d:	e8 2e fd ff ff       	call   1050 <fclose@plt>
    1322:	4c 89 ff             	mov    %r15,%rdi
    1325:	e8 06 fd ff ff       	call   1030 <free@plt>
    132a:	48 8b 7d c8          	mov    -0x38(%rbp),%rdi
    132e:	e8 fd fc ff ff       	call   1030 <free@plt>
    1333:	48 8b 7d c0          	mov    -0x40(%rbp),%rdi
    1337:	e8 f4 fc ff ff       	call   1030 <free@plt>
    133c:	31 c0                	xor    %eax,%eax
    133e:	48 83 c4 20          	add    $0x20,%rsp
    1342:	5b                   	pop    %rbx
    1343:	41 5a                	pop    %r10
    1345:	41 5c                	pop    %r12
    1347:	41 5d                	pop    %r13
    1349:	41 5e                	pop    %r14
    134b:	41 5f                	pop    %r15
    134d:	5d                   	pop    %rbp
    134e:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    1352:	c3                   	ret
    1353:	48 8d 3d d3 0c 00 00 	lea    0xcd3(%rip),%rdi        # 202d <_IO_stdin_used+0x2d>
    135a:	e8 51 fd ff ff       	call   10b0 <perror@plt>
    135f:	b8 01 00 00 00       	mov    $0x1,%eax
    1364:	eb d8                	jmp    133e <main+0x26e>
    1366:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    136d:	00 00 00 

0000000000001370 <set_fast_math>:
    1370:	f3 0f 1e fa          	endbr64
    1374:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
    1379:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
    1380:	00 
    1381:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
    1386:	c3                   	ret
    1387:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    138e:	00 00 

0000000000001390 <_start>:
    1390:	31 ed                	xor    %ebp,%ebp
    1392:	49 89 d1             	mov    %rdx,%r9
    1395:	5e                   	pop    %rsi
    1396:	48 89 e2             	mov    %rsp,%rdx
    1399:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    139d:	50                   	push   %rax
    139e:	54                   	push   %rsp
    139f:	45 31 c0             	xor    %r8d,%r8d
    13a2:	31 c9                	xor    %ecx,%ecx
    13a4:	48 8d 3d 25 fd ff ff 	lea    -0x2db(%rip),%rdi        # 10d0 <main>
    13ab:	ff 15 0f 2c 00 00    	call   *0x2c0f(%rip)        # 3fc0 <__libc_start_main@GLIBC_2.34>
    13b1:	f4                   	hlt
    13b2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    13b9:	00 00 00 
    13bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000013c0 <deregister_tm_clones>:
    13c0:	48 8d 3d 91 2c 00 00 	lea    0x2c91(%rip),%rdi        # 4058 <__TMC_END__>
    13c7:	48 8d 05 8a 2c 00 00 	lea    0x2c8a(%rip),%rax        # 4058 <__TMC_END__>
    13ce:	48 39 f8             	cmp    %rdi,%rax
    13d1:	74 15                	je     13e8 <deregister_tm_clones+0x28>
    13d3:	48 8b 05 ee 2b 00 00 	mov    0x2bee(%rip),%rax        # 3fc8 <_ITM_deregisterTMCloneTable@Base>
    13da:	48 85 c0             	test   %rax,%rax
    13dd:	74 09                	je     13e8 <deregister_tm_clones+0x28>
    13df:	ff e0                	jmp    *%rax
    13e1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    13e8:	c3                   	ret
    13e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000013f0 <register_tm_clones>:
    13f0:	48 8d 3d 61 2c 00 00 	lea    0x2c61(%rip),%rdi        # 4058 <__TMC_END__>
    13f7:	48 8d 35 5a 2c 00 00 	lea    0x2c5a(%rip),%rsi        # 4058 <__TMC_END__>
    13fe:	48 29 fe             	sub    %rdi,%rsi
    1401:	48 89 f0             	mov    %rsi,%rax
    1404:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1408:	48 c1 f8 03          	sar    $0x3,%rax
    140c:	48 01 c6             	add    %rax,%rsi
    140f:	48 d1 fe             	sar    %rsi
    1412:	74 14                	je     1428 <register_tm_clones+0x38>
    1414:	48 8b 05 bd 2b 00 00 	mov    0x2bbd(%rip),%rax        # 3fd8 <_ITM_registerTMCloneTable@Base>
    141b:	48 85 c0             	test   %rax,%rax
    141e:	74 08                	je     1428 <register_tm_clones+0x38>
    1420:	ff e0                	jmp    *%rax
    1422:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1428:	c3                   	ret
    1429:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001430 <__do_global_dtors_aux>:
    1430:	f3 0f 1e fa          	endbr64
    1434:	80 3d 2d 2c 00 00 00 	cmpb   $0x0,0x2c2d(%rip)        # 4068 <completed.0>
    143b:	75 2b                	jne    1468 <__do_global_dtors_aux+0x38>
    143d:	55                   	push   %rbp
    143e:	48 83 3d 9a 2b 00 00 	cmpq   $0x0,0x2b9a(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    1445:	00 
    1446:	48 89 e5             	mov    %rsp,%rbp
    1449:	74 0c                	je     1457 <__do_global_dtors_aux+0x27>
    144b:	48 8b 3d fe 2b 00 00 	mov    0x2bfe(%rip),%rdi        # 4050 <__dso_handle>
    1452:	e8 69 fc ff ff       	call   10c0 <__cxa_finalize@plt>
    1457:	e8 64 ff ff ff       	call   13c0 <deregister_tm_clones>
    145c:	c6 05 05 2c 00 00 01 	movb   $0x1,0x2c05(%rip)        # 4068 <completed.0>
    1463:	5d                   	pop    %rbp
    1464:	c3                   	ret
    1465:	0f 1f 00             	nopl   (%rax)
    1468:	c3                   	ret
    1469:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001470 <frame_dummy>:
    1470:	f3 0f 1e fa          	endbr64
    1474:	e9 77 ff ff ff       	jmp    13f0 <register_tm_clones>

Disassembly of section .fini:

000000000000147c <_fini>:
    147c:	48 83 ec 08          	sub    $0x8,%rsp
    1480:	48 83 c4 08          	add    $0x8,%rsp
    1484:	c3                   	ret
