import type { CSSProperties } from "react";

export type TouchSampleLike = {
  fz?: number[];
  fx?: number[];
  fy?: number[];
};

export type TouchScale = {
  normalMax: number;
  shearMax: number;
};

export type TouchLayout = {
  columns: number;
  rowLengths: number[];
  unitCount: number;
  label: string;
};

export type TouchCoordinatePoint = { index: number; xMm: number; yMm: number; zMm: number };
export type TouchShearArrow = { angleDeg: number; lengthPx: number; opacity: number };
type TouchCoordinateCellStyle = CSSProperties & {
  "--touch-arrow-angle": string;
  "--touch-arrow-color": string;
  "--touch-arrow-length": string;
  "--touch-arrow-opacity": number;
};

// Finger pad L5325 PX6AX-GEN3-CP-L5325-Omega PXSR-STDCP03A.
// Coordinates are from the vendor spreadsheet, in millimeters, with point
// numbers matching the order of the 239-element fx/fy/fz arrays.
export const PAXINI_TOUCH_POINTS = [
  { index: 1, xMm: -12.34316937, yMm: 0.08614981, zMm: 3.14987600 },
  { index: 2, xMm: -12.34320729, yMm: 4.10835492, zMm: 3.14990730 },
  { index: 3, xMm: -12.34325430, yMm: 8.93502718, zMm: 3.14992967 },
  { index: 4, xMm: -12.34329359, yMm: 12.95723576, zMm: 3.14994808 },
  { index: 5, xMm: -12.34333303, yMm: 16.97947169, zMm: 3.14996619 },
  { index: 6, xMm: -12.34338074, yMm: 21.80613744, zMm: 3.14998720 },
  { index: 7, xMm: -12.34342095, yMm: 25.82835893, zMm: 3.15000384 },
  { index: 8, xMm: -12.34346115, yMm: 29.85058027, zMm: 3.15002047 },
  { index: 9, xMm: -12.34350886, yMm: 34.67724566, zMm: 3.15004148 },
  { index: 10, xMm: -12.34354831, yMm: 38.69946673, zMm: 3.15005959 },
  { index: 11, xMm: -12.34358759, yMm: 42.72168777, zMm: 3.15007801 },
  { index: 12, xMm: -12.34363460, yMm: 47.54835306, zMm: 3.15010037 },
  { index: 13, xMm: -12.34367992, yMm: 51.57057780, zMm: 3.15012487 },
  { index: 14, xMm: -11.57816736, yMm: 1.47771477, zMm: 4.24347484 },
  { index: 15, xMm: -11.57288140, yMm: 5.32875238, zMm: 4.24926210 },
  { index: 16, xMm: -11.56652630, yMm: 9.94999750, zMm: 4.25619573 },
  { index: 17, xMm: -11.56122044, yMm: 13.80103511, zMm: 4.26196452 },
  { index: 18, xMm: -11.55590555, yMm: 17.65207389, zMm: 4.26772491 },
  { index: 19, xMm: -11.54951578, yMm: 22.27331935, zMm: 4.27462625 },
  { index: 20, xMm: -11.54418107, yMm: 26.12435723, zMm: 4.28036808 },
  { index: 21, xMm: -11.53883737, yMm: 29.97539511, zMm: 4.28610146 },
  { index: 22, xMm: -11.53241309, yMm: 34.59664057, zMm: 4.29297033 },
  { index: 23, xMm: -11.52704966, yMm: 38.44767845, zMm: 4.29868505 },
  { index: 24, xMm: -11.52167730, yMm: 42.29871634, zMm: 4.30439128 },
  { index: 25, xMm: -11.51521867, yMm: 46.91996180, zMm: 4.31122751 },
  { index: 26, xMm: -11.50940727, yMm: 50.77150704, zMm: 4.31684737 },
  { index: 27, xMm: -10.93585654, yMm: 0.08620575, zMm: 3.14992133 },
  { index: 28, xMm: -10.40703779, yMm: 1.57589339, zMm: 4.31810902 },
  { index: 29, xMm: -10.60008274, yMm: 2.57952695, zMm: 5.06320979 },
  { index: 30, xMm: -10.54685466, yMm: 6.96798547, zMm: 5.09684048 },
  { index: 31, xMm: -10.54100755, yMm: 10.69191540, zMm: 5.10051783 },
  { index: 32, xMm: -10.53531689, yMm: 14.41828248, zMm: 5.10408719 },
  { index: 33, xMm: -10.52962166, yMm: 18.14459566, zMm: 5.10764869 },
  { index: 34, xMm: -10.52278128, yMm: 22.61609779, zMm: 5.11191214 },
  { index: 35, xMm: -10.51707599, yMm: 26.34228632, zMm: 5.11545634 },
  { index: 36, xMm: -10.51136596, yMm: 30.06841650, zMm: 5.11899278 },
  { index: 37, xMm: -10.50450761, yMm: 34.53969801, zMm: 5.12322630 },
  { index: 38, xMm: -10.49878651, yMm: 38.26569443, zMm: 5.12674606 },
  { index: 39, xMm: -10.49306154, yMm: 41.99163387, zMm: 5.13025753 },
  { index: 40, xMm: -10.48618582, yMm: 46.46268383, zMm: 5.13446076 },
  { index: 41, xMm: -10.48710065, yMm: 50.17982566, zMm: 5.13451429 },
  { index: 42, xMm: -10.12685296, yMm: 51.05871000, zMm: 4.41334108 },
  { index: 43, xMm: -10.31216608, yMm: 51.80657222, zMm: 3.44195778 },
  { index: 44, xMm: -8.82488735, yMm: 0.08620506, zMm: 3.14992077 },
  { index: 45, xMm: -8.82493266, yMm: 1.80268470, zMm: 4.48935399 },
  { index: 46, xMm: -8.82488551, yMm: 3.56700720, zMm: 5.76460850 },
  { index: 47, xMm: -8.82490547, yMm: 7.17175428, zMm: 5.76463618 },
  { index: 48, xMm: -8.82492959, yMm: 11.49746902, zMm: 5.76466936 },
  { index: 49, xMm: -8.82494989, yMm: 15.10221561, zMm: 5.76469698 },
  { index: 50, xMm: -8.82497048, yMm: 18.70696903, zMm: 5.76472454 },
  { index: 51, xMm: -8.82499585, yMm: 23.03267151, zMm: 5.76475749 },
  { index: 52, xMm: -8.82501781, yMm: 26.63742436, zMm: 5.76478480 },
  { index: 53, xMm: -8.82503976, yMm: 30.24217669, zMm: 5.76481211 },
  { index: 54, xMm: -8.82506514, yMm: 34.56788982, zMm: 5.76484507 },
  { index: 55, xMm: -8.82508573, yMm: 38.17264267, zMm: 5.76487263 },
  { index: 56, xMm: -8.82510602, yMm: 41.77738577, zMm: 5.76490025 },
  { index: 57, xMm: -8.82513014, yMm: 46.10313194, zMm: 5.76493343 },
  { index: 58, xMm: -8.82515041, yMm: 49.70784448, zMm: 5.76498759 },
  { index: 59, xMm: -8.85221947, yMm: 50.97801135, zMm: 4.51490876 },
  { index: 60, xMm: -8.85347842, yMm: 51.81447517, zMm: 3.43138235 },
  { index: 61, xMm: -6.47158471, yMm: 0.08620429, zMm: 3.14992014 },
  { index: 62, xMm: -6.48972760, yMm: 2.04055889, zMm: 4.66715181 },
  { index: 63, xMm: -6.51900371, yMm: 4.05952580, zMm: 6.10414471 },
  { index: 64, xMm: -6.51622127, yMm: 7.60033981, zMm: 6.10445825 },
  { index: 65, xMm: -6.51437681, yMm: 11.85351868, zMm: 6.10466515 },
  { index: 66, xMm: -6.51284560, yMm: 15.39783249, zMm: 6.10483683 },
  { index: 67, xMm: -6.51131455, yMm: 18.94215087, zMm: 6.10500844 },
  { index: 68, xMm: -6.50947783, yMm: 23.19533577, zMm: 6.10521422 },
  { index: 69, xMm: -6.50794779, yMm: 26.73965955, zMm: 6.10538558 },
  { index: 70, xMm: -6.50641779, yMm: 30.28398532, zMm: 6.10555688 },
  { index: 71, xMm: -6.50458094, yMm: 34.53717859, zMm: 6.10576245 },
  { index: 72, xMm: -6.50305019, yMm: 38.08151668, zMm: 6.10593370 },
  { index: 73, xMm: -6.50151867, yMm: 41.62583857, zMm: 6.10610497 },
  { index: 74, xMm: -6.49967093, yMm: 45.87903249, zMm: 6.10631154 },
  { index: 75, xMm: -6.49813753, yMm: 49.42134917, zMm: 6.10647305 },
  { index: 76, xMm: -6.50024577, yMm: 50.84102182, zMm: 4.68598011 },
  { index: 77, xMm: -6.49254982, yMm: 51.82725501, zMm: 3.41426623 },
  { index: 78, xMm: -4.11828208, yMm: 0.08620352, zMm: 3.14991952 },
  { index: 79, xMm: -4.13830108, yMm: 2.18625513, zMm: 4.77514251 },
  { index: 80, xMm: -4.16379063, yMm: 4.35911602, zMm: 6.30685211 },
  { index: 81, xMm: -4.16176649, yMm: 7.86350725, zMm: 6.30697520 },
  { index: 82, xMm: -4.15990403, yMm: 12.07220646, zMm: 6.30708796 },
  { index: 83, xMm: -4.15836176, yMm: 15.57945392, zMm: 6.30718127 },
  { index: 84, xMm: -4.15681966, yMm: 19.08670293, zMm: 6.30727453 },
  { index: 85, xMm: -4.15496953, yMm: 23.29540239, zMm: 6.30738636 },
  { index: 86, xMm: -4.15342818, yMm: 26.80265241, zMm: 6.30747948 },
  { index: 87, xMm: -4.15188687, yMm: 30.30990254, zMm: 6.30757255 },
  { index: 88, xMm: -4.15003690, yMm: 34.51860494, zMm: 6.30768419 },
  { index: 89, xMm: -4.14849507, yMm: 38.02585836, zMm: 6.30777720 },
  { index: 90, xMm: -4.14695303, yMm: 41.53310108, zMm: 6.30787017 },
  { index: 91, xMm: -4.14508660, yMm: 45.74180096, zMm: 6.30798265 },
  { index: 92, xMm: -4.14352651, yMm: 49.24734355, zMm: 6.30802651 },
  { index: 93, xMm: -4.14182290, yMm: 50.76194449, zMm: 4.78397776 },
  { index: 94, xMm: -4.13162122, yMm: 51.84002086, zMm: 3.39715074 },
  { index: 95, xMm: -2.35330510, yMm: 0.08620294, zMm: 3.14991905 },
  { index: 96, xMm: -2.36655084, yMm: 2.24587558, zMm: 4.81913566 },
  { index: 97, xMm: -2.38217147, yMm: 4.48144148, zMm: 6.38883125 },
  { index: 98, xMm: -2.38118605, yMm: 7.97124805, zMm: 6.38886072 },
  { index: 99, xMm: -2.37995501, yMm: 12.16191585, zMm: 6.38889758 },
  { index: 100, xMm: -2.37893504, yMm: 15.65413803, zMm: 6.38892810 },
  { index: 101, xMm: -2.37791518, yMm: 19.14636097, zMm: 6.38895860 },
  { index: 102, xMm: -2.37669158, yMm: 23.33702875, zMm: 6.38899517 },
  { index: 103, xMm: -2.37567216, yMm: 26.82925203, zMm: 6.38902562 },
  { index: 104, xMm: -2.37465278, yMm: 30.32147525, zMm: 6.38905605 },
  { index: 105, xMm: -2.37342937, yMm: 34.51214438, zMm: 6.38909255 },
  { index: 106, xMm: -2.37240969, yMm: 38.00436912, zMm: 6.38912296 },
  { index: 107, xMm: -2.37138998, yMm: 41.49658768, zMm: 6.38915335 },
  { index: 108, xMm: -2.37015636, yMm: 45.68725519, zMm: 6.38919012 },
  { index: 109, xMm: -2.36958095, yMm: 49.17768593, zMm: 6.38933776 },
  { index: 110, xMm: -2.36848843, yMm: 50.73676980, zMm: 4.81506212 },
  { index: 111, xMm: -2.36092478, yMm: 51.84958607, zMm: 3.38431454 },
  { index: 112, xMm: -0.00000247, yMm: 0.08620157, zMm: 3.14991793 },
  { index: 113, xMm: -0.00000128, yMm: 2.27250992, zMm: 4.83875181 },
  { index: 114, xMm: -0.00000010, yMm: 4.53478151, zMm: 6.42438459 },
  { index: 115, xMm: 0.00000008, yMm: 8.02013091, zMm: 6.42436711 },
  { index: 116, xMm: 0.00000030, yMm: 12.20255018, zMm: 6.42436526 },
  { index: 117, xMm: 0.00000048, yMm: 15.68789958, zMm: 6.42436371 },
  { index: 118, xMm: 0.00000066, yMm: 19.17324897, zMm: 6.42436217 },
  { index: 119, xMm: 0.00000088, yMm: 23.35566825, zMm: 6.42436032 },
  { index: 120, xMm: 0.00000106, yMm: 26.84101764, zMm: 6.42435878 },
  { index: 121, xMm: 0.00000124, yMm: 30.32636704, zMm: 6.42435723 },
  { index: 122, xMm: 0.00000145, yMm: 34.50878631, zMm: 6.42435538 },
  { index: 123, xMm: 0.00000164, yMm: 37.99413570, zMm: 6.42435384 },
  { index: 124, xMm: 0.00000182, yMm: 41.47948510, zMm: 6.42435230 },
  { index: 125, xMm: 0.00000203, yMm: 45.66190437, zMm: 6.42435045 },
  { index: 126, xMm: 0.00000221, yMm: 49.14725377, zMm: 6.42434891 },
  { index: 127, xMm: 0.00000612, yMm: 50.74134502, zMm: 4.80941693 },
  { index: 128, xMm: 0.00000382, yMm: 51.86232744, zMm: 3.36720016 },
  { index: 129, xMm: 2.35330510, yMm: 0.08620294, zMm: 3.14991905 },
  { index: 130, xMm: 2.36655084, yMm: 2.24587558, zMm: 4.81913566 },
  { index: 131, xMm: 2.38217147, yMm: 4.48144148, zMm: 6.38883125 },
  { index: 132, xMm: 2.38118605, yMm: 7.97124805, zMm: 6.38886072 },
  { index: 133, xMm: 2.37995501, yMm: 12.16191585, zMm: 6.38889758 },
  { index: 134, xMm: 2.37893504, yMm: 15.65413803, zMm: 6.38892810 },
  { index: 135, xMm: 2.37791518, yMm: 19.14636097, zMm: 6.38895860 },
  { index: 136, xMm: 2.37669158, yMm: 23.33702875, zMm: 6.38899517 },
  { index: 137, xMm: 2.37567216, yMm: 26.82925203, zMm: 6.38902562 },
  { index: 138, xMm: 2.37465278, yMm: 30.32147525, zMm: 6.38905605 },
  { index: 139, xMm: 2.37342937, yMm: 34.51214438, zMm: 6.38909255 },
  { index: 140, xMm: 2.37240969, yMm: 38.00436912, zMm: 6.38912296 },
  { index: 141, xMm: 2.37138998, yMm: 41.49658768, zMm: 6.38915335 },
  { index: 142, xMm: 2.37015636, yMm: 45.68725519, zMm: 6.38919012 },
  { index: 143, xMm: 2.36958095, yMm: 49.17768593, zMm: 6.38933776 },
  { index: 144, xMm: 2.36848843, yMm: 50.73676980, zMm: 4.81506212 },
  { index: 145, xMm: 2.36092478, yMm: 51.84958607, zMm: 3.38431454 },
  { index: 146, xMm: 4.11828208, yMm: 0.08620352, zMm: 3.14991952 },
  { index: 147, xMm: 4.13830108, yMm: 2.18625513, zMm: 4.77514251 },
  { index: 148, xMm: 4.16379063, yMm: 4.35911602, zMm: 6.30685211 },
  { index: 149, xMm: 4.16176649, yMm: 7.86350725, zMm: 6.30697520 },
  { index: 150, xMm: 4.15990403, yMm: 12.07220646, zMm: 6.30708796 },
  { index: 151, xMm: 4.15836176, yMm: 15.57945392, zMm: 6.30718127 },
  { index: 152, xMm: 4.15681966, yMm: 19.08670293, zMm: 6.30727453 },
  { index: 153, xMm: 4.15496953, yMm: 23.29540239, zMm: 6.30738636 },
  { index: 154, xMm: 4.15342818, yMm: 26.80265241, zMm: 6.30747948 },
  { index: 155, xMm: 4.15188687, yMm: 30.30990254, zMm: 6.30757255 },
  { index: 156, xMm: 4.15003690, yMm: 34.51860494, zMm: 6.30768419 },
  { index: 157, xMm: 4.14849507, yMm: 38.02585836, zMm: 6.30777720 },
  { index: 158, xMm: 4.14695303, yMm: 41.53310108, zMm: 6.30787017 },
  { index: 159, xMm: 4.14508660, yMm: 45.74180096, zMm: 6.30798265 },
  { index: 160, xMm: 4.14352651, yMm: 49.24734355, zMm: 6.30802651 },
  { index: 161, xMm: 4.14182290, yMm: 50.76194449, zMm: 4.78397776 },
  { index: 162, xMm: 4.13162122, yMm: 51.84002086, zMm: 3.39715074 },
  { index: 163, xMm: 6.47158471, yMm: 0.08620429, zMm: 3.14992014 },
  { index: 164, xMm: 6.48972760, yMm: 2.04055889, zMm: 4.66715181 },
  { index: 165, xMm: 6.51900371, yMm: 4.05952580, zMm: 6.10414471 },
  { index: 166, xMm: 6.51622127, yMm: 7.60033981, zMm: 6.10445825 },
  { index: 167, xMm: 6.51437681, yMm: 11.85351868, zMm: 6.10466515 },
  { index: 168, xMm: 6.51284560, yMm: 15.39783249, zMm: 6.10483683 },
  { index: 169, xMm: 6.51131455, yMm: 18.94215087, zMm: 6.10500844 },
  { index: 170, xMm: 6.50947783, yMm: 23.19533577, zMm: 6.10521422 },
  { index: 171, xMm: 6.50794779, yMm: 26.73965955, zMm: 6.10538558 },
  { index: 172, xMm: 6.50641779, yMm: 30.28398532, zMm: 6.10555688 },
  { index: 173, xMm: 6.50458094, yMm: 34.53717859, zMm: 6.10576245 },
  { index: 174, xMm: 6.50305019, yMm: 38.08151668, zMm: 6.10593370 },
  { index: 175, xMm: 6.50151867, yMm: 41.62583857, zMm: 6.10610497 },
  { index: 176, xMm: 6.49967093, yMm: 45.87903249, zMm: 6.10631154 },
  { index: 177, xMm: 6.49813753, yMm: 49.42134917, zMm: 6.10647305 },
  { index: 178, xMm: 6.50024577, yMm: 50.84102182, zMm: 4.68598011 },
  { index: 179, xMm: 6.49254982, yMm: 51.82725501, zMm: 3.41426623 },
  { index: 180, xMm: 8.82488735, yMm: 0.08620506, zMm: 3.14992077 },
  { index: 181, xMm: 8.82493266, yMm: 1.80268470, zMm: 4.48935399 },
  { index: 182, xMm: 8.82488551, yMm: 3.56700720, zMm: 5.76460850 },
  { index: 183, xMm: 8.82490547, yMm: 7.17175428, zMm: 5.76463618 },
  { index: 184, xMm: 8.82492959, yMm: 11.49746902, zMm: 5.76466936 },
  { index: 185, xMm: 8.82494989, yMm: 15.10221561, zMm: 5.76469698 },
  { index: 186, xMm: 8.82497048, yMm: 18.70696903, zMm: 5.76472454 },
  { index: 187, xMm: 8.82499585, yMm: 23.03267151, zMm: 5.76475749 },
  { index: 188, xMm: 8.82501781, yMm: 26.63742436, zMm: 5.76478480 },
  { index: 189, xMm: 8.82503976, yMm: 30.24217669, zMm: 5.76481211 },
  { index: 190, xMm: 8.82506514, yMm: 34.56788982, zMm: 5.76484507 },
  { index: 191, xMm: 8.82508573, yMm: 38.17264267, zMm: 5.76487263 },
  { index: 192, xMm: 8.82510602, yMm: 41.77738577, zMm: 5.76490025 },
  { index: 193, xMm: 8.82513014, yMm: 46.10313194, zMm: 5.76493343 },
  { index: 194, xMm: 8.82515041, yMm: 49.70784448, zMm: 5.76498759 },
  { index: 195, xMm: 8.85221947, yMm: 50.97801135, zMm: 4.51490876 },
  { index: 196, xMm: 8.85347842, yMm: 51.81447517, zMm: 3.43138235 },
  { index: 197, xMm: 10.93585654, yMm: 0.08620575, zMm: 3.14992133 },
  { index: 198, xMm: 10.40703779, yMm: 1.57589339, zMm: 4.31810902 },
  { index: 199, xMm: 10.60008274, yMm: 2.57952695, zMm: 5.06320979 },
  { index: 200, xMm: 10.54685466, yMm: 6.96798547, zMm: 5.09684048 },
  { index: 201, xMm: 10.54100755, yMm: 10.69191540, zMm: 5.10051783 },
  { index: 202, xMm: 10.53531689, yMm: 14.41828248, zMm: 5.10408719 },
  { index: 203, xMm: 10.52962166, yMm: 18.14459566, zMm: 5.10764869 },
  { index: 204, xMm: 10.52278128, yMm: 22.61609779, zMm: 5.11191214 },
  { index: 205, xMm: 10.51707599, yMm: 26.34228632, zMm: 5.11545634 },
  { index: 206, xMm: 10.51136596, yMm: 30.06841650, zMm: 5.11899278 },
  { index: 207, xMm: 10.50450761, yMm: 34.53969801, zMm: 5.12322630 },
  { index: 208, xMm: 10.49878651, yMm: 38.26569443, zMm: 5.12674606 },
  { index: 209, xMm: 10.49306154, yMm: 41.99163387, zMm: 5.13025753 },
  { index: 210, xMm: 10.48618582, yMm: 46.46268383, zMm: 5.13446076 },
  { index: 211, xMm: 10.48710065, yMm: 50.17982566, zMm: 5.13451429 },
  { index: 212, xMm: 10.12685296, yMm: 51.05871000, zMm: 4.41334108 },
  { index: 213, xMm: 10.31216608, yMm: 51.80657222, zMm: 3.44195778 },
  { index: 214, xMm: 11.57816736, yMm: 1.47771477, zMm: 4.24347484 },
  { index: 215, xMm: 11.57288140, yMm: 5.32875238, zMm: 4.24926210 },
  { index: 216, xMm: 11.56652630, yMm: 9.94999750, zMm: 4.25619573 },
  { index: 217, xMm: 11.56122044, yMm: 13.80103511, zMm: 4.26196452 },
  { index: 218, xMm: 11.55590555, yMm: 17.65207389, zMm: 4.26772491 },
  { index: 219, xMm: 11.54951578, yMm: 22.27331935, zMm: 4.27462625 },
  { index: 220, xMm: 11.54418107, yMm: 26.12435723, zMm: 4.28036808 },
  { index: 221, xMm: 11.53883737, yMm: 29.97539511, zMm: 4.28610146 },
  { index: 222, xMm: 11.53241309, yMm: 34.59664057, zMm: 4.29297033 },
  { index: 223, xMm: 11.52704966, yMm: 38.44767845, zMm: 4.29868505 },
  { index: 224, xMm: 11.52167730, yMm: 42.29871634, zMm: 4.30439128 },
  { index: 225, xMm: 11.51521867, yMm: 46.91996180, zMm: 4.31122751 },
  { index: 226, xMm: 11.50940727, yMm: 50.77150704, zMm: 4.31684737 },
  { index: 227, xMm: 12.34316937, yMm: 0.08614981, zMm: 3.14987600 },
  { index: 228, xMm: 12.34320729, yMm: 4.10835492, zMm: 3.14990730 },
  { index: 229, xMm: 12.34325430, yMm: 8.93502718, zMm: 3.14992967 },
  { index: 230, xMm: 12.34329359, yMm: 12.95723576, zMm: 3.14994808 },
  { index: 231, xMm: 12.34333303, yMm: 16.97947169, zMm: 3.14996619 },
  { index: 232, xMm: 12.34338074, yMm: 21.80613744, zMm: 3.14998720 },
  { index: 233, xMm: 12.34342095, yMm: 25.82835893, zMm: 3.15000384 },
  { index: 234, xMm: 12.34346115, yMm: 29.85058027, zMm: 3.15002047 },
  { index: 235, xMm: 12.34350886, yMm: 34.67724566, zMm: 3.15004148 },
  { index: 236, xMm: 12.34354831, yMm: 38.69946673, zMm: 3.15005959 },
  { index: 237, xMm: 12.34358759, yMm: 42.72168777, zMm: 3.15007801 },
  { index: 238, xMm: 12.34363460, yMm: 47.54835306, zMm: 3.15010037 },
  { index: 239, xMm: 12.34367992, yMm: 51.57057780, zMm: 3.15012487 },
] as const satisfies readonly TouchCoordinatePoint[];

export const PAXINI_TOUCH_UNIT_COUNT = PAXINI_TOUCH_POINTS.length;
const PAXINI_TOUCH_X_MIN = Math.min(...PAXINI_TOUCH_POINTS.map((point) => point.xMm));
const PAXINI_TOUCH_X_MAX = Math.max(...PAXINI_TOUCH_POINTS.map((point) => point.xMm));
const PAXINI_TOUCH_Y_MIN = Math.min(...PAXINI_TOUCH_POINTS.map((point) => point.yMm));
const PAXINI_TOUCH_Y_MAX = Math.max(...PAXINI_TOUCH_POINTS.map((point) => point.yMm));
const PAXINI_TOUCH_WIDTH_MM = Math.max(PAXINI_TOUCH_X_MAX - PAXINI_TOUCH_X_MIN, 1);
const PAXINI_TOUCH_HEIGHT_MM = Math.max(PAXINI_TOUCH_Y_MAX - PAXINI_TOUCH_Y_MIN, 1);
const PAXINI_TOUCH_POINT_INSET_PCT = 5;

const FALLBACK_TOUCH_COLUMNS = 16;

export function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

function rectangularLayout(rows: number, columns: number, label: string): TouchLayout {
  return {
    columns,
    rowLengths: Array.from({ length: rows }, () => columns),
    unitCount: rows * columns,
    label,
  };
}

export function touchLayoutForCount(unitCount: number): TouchLayout {
  if (unitCount === PAXINI_TOUCH_UNIT_COUNT) {
    return {
      columns: 0,
      rowLengths: [],
      unitCount: PAXINI_TOUCH_UNIT_COUNT,
      label: `${PAXINI_TOUCH_UNIT_COUNT} taxels · XYZ map`,
    };
  }
  if (unitCount === 500) {
    return rectangularLayout(50, 10, "50 x 10");
  }
  if (unitCount > 0 && unitCount % 10 === 0 && unitCount / 10 >= 10) {
    return rectangularLayout(unitCount / 10, 10, `${unitCount / 10} x 10`);
  }
  const columns = Math.min(FALLBACK_TOUCH_COLUMNS, Math.max(1, Math.ceil(Math.sqrt(Math.max(unitCount, 1)))));
  const rows = Math.ceil(unitCount / columns);
  const rowLengths = Array.from({ length: rows }, (_, row) => {
    const remaining = unitCount - row * columns;
    return Math.max(0, Math.min(columns, remaining));
  }).filter((length) => length > 0);
  return {
    columns,
    rowLengths,
    unitCount,
    label: `${unitCount} taxels`,
  };
}

function interpolateChannel(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

export function normalTouchColor(value: number, scaleMax: number): string {
  const stops = [
    [17, 24, 39],
    [37, 99, 235],
    [20, 184, 166],
    [250, 204, 21],
    [239, 68, 68],
  ];
  const normalized = Math.max(0, Math.min(1, Math.abs(value) / Math.max(scaleMax, 1)));
  const scaled = normalized * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const t = scaled - index;
  const a = stops[index];
  const b = stops[index + 1];
  return `rgb(${interpolateChannel(a[0], b[0], t)}, ${interpolateChannel(a[1], b[1], t)}, ${interpolateChannel(a[2], b[2], t)})`;
}

export function touchScaleFromSamples(samples: Array<TouchSampleLike | null | undefined>): TouchScale {
  let normalMax = 1;
  let shearMax = 1;
  for (const sample of samples) {
    const fz = sample?.fz ?? [];
    const fx = sample?.fx ?? [];
    const fy = sample?.fy ?? [];
    for (const value of fz) {
      if (Number.isFinite(value)) {
        normalMax = Math.max(normalMax, Math.abs(value));
      }
    }
    const count = Math.max(fx.length, fy.length);
    for (let index = 0; index < count; index += 1) {
      const shear = Math.hypot(fx[index] ?? 0, fy[index] ?? 0);
      if (Number.isFinite(shear)) {
        shearMax = Math.max(shearMax, shear);
      }
    }
  }
  return { normalMax, shearMax };
}

export function touchCellColor(fz: number, fx: number | undefined, fy: number | undefined, scale: TouchScale): string {
  const shear = Math.hypot(fx ?? 0, fy ?? 0);
  if (shear > 0) {
    const shearRatio = Math.max(0, Math.min(1, shear / Math.max(scale.shearMax, 1)));
    const normalRatio = Math.max(0, Math.min(1, Math.abs(fz) / Math.max(scale.normalMax, 1)));
    const hue = (Math.atan2(fy ?? 0, fx ?? 0) * 180 / Math.PI + 360) % 360;
    const saturation = 42 + shearRatio * 48;
    const lightness = 18 + normalRatio * 38 + shearRatio * 10;
    return `hsl(${hue.toFixed(0)}deg ${saturation.toFixed(0)}% ${lightness.toFixed(0)}%)`;
  }
  return normalTouchColor(fz, scale.normalMax);
}

export function touchSampleHasShear(sample: TouchSampleLike | null | undefined): boolean {
  const fx = sample?.fx ?? [];
  const fy = sample?.fy ?? [];
  const count = Math.max(fx.length, fy.length);
  for (let index = 0; index < count; index += 1) {
    if (Math.hypot(fx[index] ?? 0, fy[index] ?? 0) > 0) {
      return true;
    }
  }
  return false;
}

export function touchSampleActivePoints(sample: TouchSampleLike | null | undefined): number {
  const fz = sample?.fz ?? [];
  const fx = sample?.fx ?? [];
  const fy = sample?.fy ?? [];
  const count = Math.max(fz.length, fx.length, fy.length);
  let active = 0;
  for (let index = 0; index < count; index += 1) {
    if (Math.abs(fz[index] ?? 0) > 0 || Math.abs(fx[index] ?? 0) > 0 || Math.abs(fy[index] ?? 0) > 0) {
      active += 1;
    }
  }
  return active;
}

export function touchSampleLocalMax(sample: TouchSampleLike | null | undefined): number {
  return Math.max(0, ...(sample?.fz ?? []).map((value) => Math.abs(value)));
}

export function touchShearArrow(fx: number | undefined, fy: number | undefined, scale: TouchScale): TouchShearArrow {
  const shearX = fx ?? 0;
  const shearY = fy ?? 0;
  const shear = Math.hypot(shearX, shearY);
  if (shear <= 0 || !Number.isFinite(shear)) {
    return { angleDeg: 0, lengthPx: 0, opacity: 0 };
  }
  const normalized = Math.max(0, Math.min(1, shear / Math.max(scale.shearMax, 1)));
  return {
    angleDeg: Math.atan2(-shearY, shearX) * 180 / Math.PI,
    lengthPx: 4 + normalized * 20,
    opacity: 0.92,
  };
}

function touchCoordinatePointsForCount(unitCount: number): readonly TouchCoordinatePoint[] | null {
  return unitCount === PAXINI_TOUCH_UNIT_COUNT ? PAXINI_TOUCH_POINTS : null;
}

function coordinatePct(value: number, min: number, span: number): number {
  const raw = (value - min) / span;
  const normalized = Math.max(0, Math.min(1, raw));
  return PAXINI_TOUCH_POINT_INSET_PCT + normalized * (100 - PAXINI_TOUCH_POINT_INSET_PCT * 2);
}

export function TouchHeatmapGrid({
  sample,
  scale,
  ariaLabel,
  emptyText = "no touch sample",
  className = "touch-grid",
}: {
  sample?: TouchSampleLike | null;
  scale?: TouchScale;
  ariaLabel: string;
  emptyText?: string;
  className?: string;
}) {
  const fz = sample?.fz ?? [];
  if (fz.length === 0) {
    return <div className="touch-empty">{emptyText}</div>;
  }
  const fx = sample?.fx ?? [];
  const fy = sample?.fy ?? [];
  const effectiveScale = scale ?? touchScaleFromSamples([sample]);
  const coordinatePoints = touchCoordinatePointsForCount(fz.length);
  if (coordinatePoints) {
    return (
      <div
        className={`${className} touch-coordinate-grid`}
        aria-label={ariaLabel}
        style={{ aspectRatio: `${PAXINI_TOUCH_WIDTH_MM} / ${PAXINI_TOUCH_HEIGHT_MM}` }}
      >
        {coordinatePoints.map((point, index) => {
          const value = fz[index] ?? 0;
          const shearX = fx[index];
          const shearY = fy[index];
          const shear = Math.hypot(shearX ?? 0, shearY ?? 0);
          const color = touchCellColor(value, shearX, shearY, effectiveScale);
          const arrow = touchShearArrow(shearX, shearY, effectiveScale);
          const cellStyle: TouchCoordinateCellStyle = {
            "--touch-arrow-angle": `${arrow.angleDeg}deg`,
            "--touch-arrow-color": color,
            "--touch-arrow-length": `${arrow.lengthPx}px`,
            "--touch-arrow-opacity": arrow.opacity,
            backgroundColor: color,
            left: `${coordinatePct(point.xMm, PAXINI_TOUCH_X_MIN, PAXINI_TOUCH_WIDTH_MM)}%`,
            top: `${100 - coordinatePct(point.yMm, PAXINI_TOUCH_Y_MIN, PAXINI_TOUCH_HEIGHT_MM)}%`,
          };
          const forceText = shear > 0
            ? `fx=${(shearX ?? 0).toFixed(1)} fy=${(shearY ?? 0).toFixed(1)} fz=${value.toFixed(1)} (0.1N)`
            : `fz=${value.toFixed(1)} (0.1N)`;
          return (
            <span
              className="touch-cell touch-coordinate-cell"
              key={point.index}
              title={`#${point.index} x=${point.xMm.toFixed(2)}mm y=${point.yMm.toFixed(2)}mm z=${point.zMm.toFixed(2)}mm · ${forceText}`}
              style={cellStyle}
            >
              <span className="touch-shear-arrow" aria-hidden="true" />
            </span>
          );
        })}
      </div>
    );
  }

  const layout = touchLayoutForCount(fz.length);
  let cursor = 0;

  return (
    <div className={className} aria-label={ariaLabel} style={{ gridTemplateRows: `repeat(${layout.rowLengths.length}, minmax(0, 1fr))` }}>
      {layout.rowLengths.map((length, rowIndex) => {
        const offset = Math.floor((layout.columns - length) / 2);
        const row = fz.slice(cursor, cursor + length);
        const startIndex = cursor;
        cursor += length;
        return (
          <div className="touch-row" key={rowIndex} style={{ gridTemplateColumns: `repeat(${layout.columns}, minmax(0, 1fr))` }}>
            {Array.from({ length: offset }).map((_, index) => (
              <span className="touch-cell touch-cell-empty" key={`pre-${index}`} />
            ))}
            {row.map((value, index) => {
              const pointIndex = startIndex + index + 1;
              const shearX = fx[startIndex + index];
              const shearY = fy[startIndex + index];
              const shear = Math.hypot(shearX ?? 0, shearY ?? 0);
              const title = shear > 0
                ? `#${pointIndex} fx=${(shearX ?? 0).toFixed(1)} fy=${(shearY ?? 0).toFixed(1)} fz=${value.toFixed(1)} (0.1N)`
                : `#${pointIndex} fz=${value.toFixed(1)} (0.1N)`;
              return (
                <span
                  className="touch-cell"
                  key={pointIndex}
                  title={title}
                  style={{ backgroundColor: touchCellColor(value, shearX, shearY, effectiveScale) }}
                />
              );
            })}
            {Array.from({ length: layout.columns - length - offset }).map((_, index) => (
              <span className="touch-cell touch-cell-empty" key={`post-${index}`} />
            ))}
          </div>
        );
      })}
    </div>
  );
}
