let done = 0;
const regionNames = ["R1", "R2", "R3", "R7", "R8"];
const subProcesses = regionNames
	.map(
		(rn) =>
			`https://www.iarai.ac.at/wp-content/uploads/sites/3/w4c21-download/IEEE_BD/competition/ieee-bd-core/${rn}.zip`
	)
	.map((url) => Deno.run({ cmd: ["wget", url] }).status());

let i = 0;
for await (const subProcess of subProcesses) {
	const region = regionNames[i];
	console.log(`Unzipping region {region}`);
	await Deno.run({ cmd: ["mkdir", region] })
	await Deno.run({ cmd: ["unzip", `${region}.zip`, "-d", region] })
	i += 1;
}

console.log("All done");
