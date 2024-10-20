import openpyxl

wb = openpyxl.load_workbook("input_data.xlsx")
ws = wb.active
data_rows = range(10, 4)
for row in data_rows:
    ws.cell(row=row, column=3, value=f"=A{row}+B{row}")

wb.save("output_data.xlsx")

print("Formulas have been successfully applied to output_data.xlsx")
