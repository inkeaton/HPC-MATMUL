
bin/seq_bench/gcc/matmul_O2:     file format elf64-x86-64


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
    10ed:	49 89 c7             	mov    %rax,%r15
    10f0:	e8 9b ff ff ff       	call   1090 <malloc@plt>
    10f5:	be 01 00 00 00       	mov    $0x1,%esi
    10fa:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
    10ff:	48 89 c3             	mov    %rax,%rbx
    1102:	48 89 04 24          	mov    %rax,(%rsp)
    1106:	e8 65 ff ff ff       	call   1070 <calloc@plt>
    110b:	f2 0f 10 0d 2d 0f 00 	movsd  0xf2d(%rip),%xmm1        # 2040 <_IO_stdin_used+0x40>
    1112:	00 
    1113:	48 89 d9             	mov    %rbx,%rcx
    1116:	4c 89 fa             	mov    %r15,%rdx
    1119:	f2 0f 10 05 27 0f 00 	movsd  0xf27(%rip),%xmm0        # 2048 <_IO_stdin_used+0x48>
    1120:	00 
    1121:	49 89 c5             	mov    %rax,%r13
    1124:	49 8d b7 00 c2 eb 0b 	lea    0xbebc200(%r15),%rsi
    112b:	66 0f 14 c9          	unpcklpd %xmm1,%xmm1
    112f:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
    1133:	31 c0                	xor    %eax,%eax
    1135:	0f 11 0c 02          	movups %xmm1,(%rdx,%rax,1)
    1139:	0f 11 04 01          	movups %xmm0,(%rcx,%rax,1)
    113d:	48 83 c0 10          	add    $0x10,%rax
    1141:	48 3d 40 9c 00 00    	cmp    $0x9c40,%rax
    1147:	75 ec                	jne    1135 <main+0x65>
    1149:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
    1150:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    1157:	48 39 f2             	cmp    %rsi,%rdx
    115a:	75 d7                	jne    1133 <main+0x63>
    115c:	e8 df fe ff ff       	call   1040 <clock@plt>
    1161:	45 31 c0             	xor    %r8d,%r8d
    1164:	48 89 c3             	mov    %rax,%rbx
    1167:	48 8b 04 24          	mov    (%rsp),%rax
    116b:	48 8d b8 00 c2 eb 0b 	lea    0xbebc200(%rax),%rdi
    1172:	48 8b 0c 24          	mov    (%rsp),%rcx
    1176:	4b 8d 34 07          	lea    (%r15,%r8,1),%rsi
    117a:	4b 8d 14 28          	lea    (%r8,%r13,1),%rdx
    117e:	66 90                	xchg   %ax,%ax
    1180:	f2 0f 10 0e          	movsd  (%rsi),%xmm1
    1184:	31 c0                	xor    %eax,%eax
    1186:	66 0f 14 c9          	unpcklpd %xmm1,%xmm1
    118a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1190:	66 0f 10 04 01       	movupd (%rcx,%rax,1),%xmm0
    1195:	66 0f 10 14 02       	movupd (%rdx,%rax,1),%xmm2
    119a:	66 0f 59 c1          	mulpd  %xmm1,%xmm0
    119e:	66 0f 58 c2          	addpd  %xmm2,%xmm0
    11a2:	0f 11 04 02          	movups %xmm0,(%rdx,%rax,1)
    11a6:	48 83 c0 10          	add    $0x10,%rax
    11aa:	48 3d 40 9c 00 00    	cmp    $0x9c40,%rax
    11b0:	75 de                	jne    1190 <main+0xc0>
    11b2:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
    11b9:	48 83 c6 08          	add    $0x8,%rsi
    11bd:	48 39 f9             	cmp    %rdi,%rcx
    11c0:	75 be                	jne    1180 <main+0xb0>
    11c2:	49 81 c0 40 9c 00 00 	add    $0x9c40,%r8
    11c9:	49 81 f8 00 c2 eb 0b 	cmp    $0xbebc200,%r8
    11d0:	75 a0                	jne    1172 <main+0xa2>
    11d2:	e8 69 fe ff ff       	call   1040 <clock@plt>
    11d7:	48 8b 3d 82 2e 00 00 	mov    0x2e82(%rip),%rdi        # 4060 <stderr@GLIBC_2.2.5>
    11de:	66 0f ef c0          	pxor   %xmm0,%xmm0
    11e2:	ba 88 13 00 00       	mov    $0x1388,%edx
    11e7:	48 29 d8             	sub    %rbx,%rax
    11ea:	48 8d 35 13 0e 00 00 	lea    0xe13(%rip),%rsi        # 2004 <_IO_stdin_used+0x4>
    11f1:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    11f6:	b8 01 00 00 00       	mov    $0x1,%eax
    11fb:	f2 0f 5e 05 4d 0e 00 	divsd  0xe4d(%rip),%xmm0        # 2050 <_IO_stdin_used+0x50>
    1202:	00 
    1203:	e8 78 fe ff ff       	call   1080 <fprintf@plt>
    1208:	48 8d 35 10 0e 00 00 	lea    0xe10(%rip),%rsi        # 201f <_IO_stdin_used+0x1f>
    120f:	48 8d 3d 0b 0e 00 00 	lea    0xe0b(%rip),%rdi        # 2021 <_IO_stdin_used+0x21>
    1216:	e8 85 fe ff ff       	call   10a0 <fopen@plt>
    121b:	48 89 c3             	mov    %rax,%rbx
    121e:	48 85 c0             	test   %rax,%rax
    1221:	0f 84 a2 00 00 00    	je     12c9 <main+0x1f9>
    1227:	48 89 c7             	mov    %rax,%rdi
    122a:	ba 88 13 00 00       	mov    $0x1388,%edx
    122f:	31 c0                	xor    %eax,%eax
    1231:	48 8d 35 fb 0d 00 00 	lea    0xdfb(%rip),%rsi        # 2033 <_IO_stdin_used+0x33>
    1238:	49 8d ad 40 1f 00 00 	lea    0x1f40(%r13),%rbp
    123f:	e8 3c fe ff ff       	call   1080 <fprintf@plt>
    1244:	49 8d 85 40 79 62 02 	lea    0x2627940(%r13),%rax
    124b:	4c 8d 25 e6 0d 00 00 	lea    0xde6(%rip),%r12        # 2038 <_IO_stdin_used+0x38>
    1252:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    1257:	4c 8d b5 c0 e0 ff ff 	lea    -0x1f40(%rbp),%r14
    125e:	f2 41 0f 10 06       	movsd  (%r14),%xmm0
    1263:	4c 89 e6             	mov    %r12,%rsi
    1266:	48 89 df             	mov    %rbx,%rdi
    1269:	b8 01 00 00 00       	mov    $0x1,%eax
    126e:	49 83 c6 08          	add    $0x8,%r14
    1272:	e8 09 fe ff ff       	call   1080 <fprintf@plt>
    1277:	4c 39 f5             	cmp    %r14,%rbp
    127a:	75 e2                	jne    125e <main+0x18e>
    127c:	48 89 de             	mov    %rbx,%rsi
    127f:	bf 0a 00 00 00       	mov    $0xa,%edi
    1284:	48 81 c5 40 9c 00 00 	add    $0x9c40,%rbp
    128b:	e8 d0 fd ff ff       	call   1060 <fputc@plt>
    1290:	48 39 6c 24 08       	cmp    %rbp,0x8(%rsp)
    1295:	75 c0                	jne    1257 <main+0x187>
    1297:	48 89 df             	mov    %rbx,%rdi
    129a:	e8 b1 fd ff ff       	call   1050 <fclose@plt>
    129f:	4c 89 ff             	mov    %r15,%rdi
    12a2:	e8 89 fd ff ff       	call   1030 <free@plt>
    12a7:	48 8b 3c 24          	mov    (%rsp),%rdi
    12ab:	e8 80 fd ff ff       	call   1030 <free@plt>
    12b0:	4c 89 ef             	mov    %r13,%rdi
    12b3:	e8 78 fd ff ff       	call   1030 <free@plt>
    12b8:	31 c0                	xor    %eax,%eax
    12ba:	48 83 c4 18          	add    $0x18,%rsp
    12be:	5b                   	pop    %rbx
    12bf:	5d                   	pop    %rbp
    12c0:	41 5c                	pop    %r12
    12c2:	41 5d                	pop    %r13
    12c4:	41 5e                	pop    %r14
    12c6:	41 5f                	pop    %r15
    12c8:	c3                   	ret
    12c9:	48 8d 3d 5d 0d 00 00 	lea    0xd5d(%rip),%rdi        # 202d <_IO_stdin_used+0x2d>
    12d0:	e8 db fd ff ff       	call   10b0 <perror@plt>
    12d5:	b8 01 00 00 00       	mov    $0x1,%eax
    12da:	eb de                	jmp    12ba <main+0x1ea>
    12dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000012e0 <_start>:
    12e0:	31 ed                	xor    %ebp,%ebp
    12e2:	49 89 d1             	mov    %rdx,%r9
    12e5:	5e                   	pop    %rsi
    12e6:	48 89 e2             	mov    %rsp,%rdx
    12e9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    12ed:	50                   	push   %rax
    12ee:	54                   	push   %rsp
    12ef:	45 31 c0             	xor    %r8d,%r8d
    12f2:	31 c9                	xor    %ecx,%ecx
    12f4:	48 8d 3d d5 fd ff ff 	lea    -0x22b(%rip),%rdi        # 10d0 <main>
    12fb:	ff 15 bf 2c 00 00    	call   *0x2cbf(%rip)        # 3fc0 <__libc_start_main@GLIBC_2.34>
    1301:	f4                   	hlt
    1302:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    1309:	00 00 00 
    130c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001310 <deregister_tm_clones>:
    1310:	48 8d 3d 41 2d 00 00 	lea    0x2d41(%rip),%rdi        # 4058 <__TMC_END__>
    1317:	48 8d 05 3a 2d 00 00 	lea    0x2d3a(%rip),%rax        # 4058 <__TMC_END__>
    131e:	48 39 f8             	cmp    %rdi,%rax
    1321:	74 15                	je     1338 <deregister_tm_clones+0x28>
    1323:	48 8b 05 9e 2c 00 00 	mov    0x2c9e(%rip),%rax        # 3fc8 <_ITM_deregisterTMCloneTable@Base>
    132a:	48 85 c0             	test   %rax,%rax
    132d:	74 09                	je     1338 <deregister_tm_clones+0x28>
    132f:	ff e0                	jmp    *%rax
    1331:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1338:	c3                   	ret
    1339:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001340 <register_tm_clones>:
    1340:	48 8d 3d 11 2d 00 00 	lea    0x2d11(%rip),%rdi        # 4058 <__TMC_END__>
    1347:	48 8d 35 0a 2d 00 00 	lea    0x2d0a(%rip),%rsi        # 4058 <__TMC_END__>
    134e:	48 29 fe             	sub    %rdi,%rsi
    1351:	48 89 f0             	mov    %rsi,%rax
    1354:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1358:	48 c1 f8 03          	sar    $0x3,%rax
    135c:	48 01 c6             	add    %rax,%rsi
    135f:	48 d1 fe             	sar    %rsi
    1362:	74 14                	je     1378 <register_tm_clones+0x38>
    1364:	48 8b 05 6d 2c 00 00 	mov    0x2c6d(%rip),%rax        # 3fd8 <_ITM_registerTMCloneTable@Base>
    136b:	48 85 c0             	test   %rax,%rax
    136e:	74 08                	je     1378 <register_tm_clones+0x38>
    1370:	ff e0                	jmp    *%rax
    1372:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1378:	c3                   	ret
    1379:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001380 <__do_global_dtors_aux>:
    1380:	f3 0f 1e fa          	endbr64
    1384:	80 3d dd 2c 00 00 00 	cmpb   $0x0,0x2cdd(%rip)        # 4068 <completed.0>
    138b:	75 2b                	jne    13b8 <__do_global_dtors_aux+0x38>
    138d:	55                   	push   %rbp
    138e:	48 83 3d 4a 2c 00 00 	cmpq   $0x0,0x2c4a(%rip)        # 3fe0 <__cxa_finalize@GLIBC_2.2.5>
    1395:	00 
    1396:	48 89 e5             	mov    %rsp,%rbp
    1399:	74 0c                	je     13a7 <__do_global_dtors_aux+0x27>
    139b:	48 8b 3d ae 2c 00 00 	mov    0x2cae(%rip),%rdi        # 4050 <__dso_handle>
    13a2:	e8 19 fd ff ff       	call   10c0 <__cxa_finalize@plt>
    13a7:	e8 64 ff ff ff       	call   1310 <deregister_tm_clones>
    13ac:	c6 05 b5 2c 00 00 01 	movb   $0x1,0x2cb5(%rip)        # 4068 <completed.0>
    13b3:	5d                   	pop    %rbp
    13b4:	c3                   	ret
    13b5:	0f 1f 00             	nopl   (%rax)
    13b8:	c3                   	ret
    13b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000013c0 <frame_dummy>:
    13c0:	f3 0f 1e fa          	endbr64
    13c4:	e9 77 ff ff ff       	jmp    1340 <register_tm_clones>

Disassembly of section .fini:

00000000000013cc <_fini>:
    13cc:	48 83 ec 08          	sub    $0x8,%rsp
    13d0:	48 83 c4 08          	add    $0x8,%rsp
    13d4:	c3                   	ret
