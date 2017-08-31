-- Ce fichier définie des fonctions utiles pour le système
require 'optim'

-- Arguments : une fonction feval, un nombre quelconque de paramètres
-- Résultats : Exécute la fonction feval avec les paramètres et affiche le temps d'execution sur la sortie standard
function time(feval, ... )
	local time = sys.clock()
	local res = {feval(unpack({...}))}
	time = sys.clock() - time
	print(string.format('\tTime : %s', xlua.formatTime(time)))
	return unpack(res)
end

-- Arguments : une matrice de confusion
-- Résultats : retourne les différentes accuracy (in french?) souhaitées
function get_accuracy(confusion)
	confusion:updateValids()

	local avg_row = (confusion.averageValid*100)
	local avg_voc = (confusion.averageUnionValid*100)
	local glb_cor = (confusion.totalValid*100)

	---[[
	local nclasses = confusion.nclasses
	for t=1, nclasses do
		local pclass = confusion.valids[t] * 100
		pclass = string.format('%06.3f', pclass)
		if confusion.classes and confusion.classes[1] then
			print(pclass .. '% [class: ' .. (confusion.classes[t] or '') .. ']')
		else
			print(pclass .. '%')
		end
	end

	print(' + average row correct: ' .. avg_row .. '%')
	print(' + average rowUcol correct (VOC measure): ' .. avg_voc .. '%')
	print(' + global correct: ' .. glb_cor .. '%')
	print('')
	--]]
	--[[
			print(confusion)
	--]]
	
	return avg_row, avg_voc, glb_cor
end
